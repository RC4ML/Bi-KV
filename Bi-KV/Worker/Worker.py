from ast import List
from functools import cache
import time
from wsgiref.util import request_uri
from httpx import request
import torch.distributed.rpc as rpc
import torch.distributed as dist
from wandb import finish
from inputGenerator.inputGenerator import InputPrompt
from rpc_def import *
from DistributedStorage.CacheCoordinator import CacheCoordinator
from DistributedStorage.Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
from Remote.remote_call import call_remote_method
import torch
from config import *
from Model.qwen2 import process_task_info, prepare_attention_meta, Qwen2ForCausalLM, AttentionMetadata
from transformers import Qwen2Config
from vllm.config import CacheConfig

import time

# TODO 提取model_params作为公共参数
model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

token_shape = (model_params['head_size'],
               model_params['num_kv_heads'],
               model_params['num_layers'],
               2)

class Worker:
    def __init__(self, rank, coordinator_rref):
        self.rank = rank
        self.worker_index=rank-WORKER_offset
        self.coordinator_rref = coordinator_rref
        self.gpu_index = 0 # NOTE: set to rank in a single machine
        self.device = torch.device(f"cuda:{self.gpu_index}")
        self.buffer_control_dict = {}
        self.buffer_size = 1000
        self.compute_buffer = torch.full(
            (self.buffer_size,) + token_shape, 
            self.rank,
            device='cpu',
            dtype=torch.float16
        )
        self.start_pos = 0
        self.cache_miss_dict = {}
        
        ## initialize model and parameters for inference
        intermediate_size = 8960
        head_dim = 128
        kv_cache_block_size = 16
        max_kv_cache_blocks = 10240        
        self.local_kv_cache_block_size = kv_cache_block_size
        self.local_max_kv_cache_blocks = max_kv_cache_blocks
        self.local_kvcache = [
            torch.randn(
            self.local_max_kv_cache_blocks * self.local_kv_cache_block_size, 2, model_params['num_kv_heads'], head_dim, dtype=torch.float16, device=self.device
            ) for _ in range(model_params['num_layers'])
        ]
        print(self.local_kvcache[0].shape)
        self.cache_config = CacheConfig(kv_cache_block_size, 1.0, 1, "auto")
        self.model_config = Qwen2Config(hidden_size = model_params['num_q_heads']*head_dim,
                                        intermediate_size = intermediate_size,
                                        num_hidden_layers = model_params['num_layers'],
                                        num_attention_heads = model_params['num_q_heads'],
                                        num_key_value_heads = model_params['num_kv_heads'])
        torch.set_default_dtype(torch.float16)
        self.model = Qwen2ForCausalLM(self.device, self.model_config, self.cache_config).to(self.device)
        print(f"[Worker][RANK {self.rank}] Init Worker")

    def forward(self, task_info_list:List):
        coordinator_owner = self.coordinator_rref.owner()
        req_id = task_info_list[0]['request_id']
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Add {len(task_info_list)} requests to coordinator")
        rpc.rpc_sync(to=coordinator_owner, 
                         func=call_remote_method, 
                         args=(CacheCoordinator.add_requests,self.coordinator_rref, 
                               task_info_list))
        finished_signal = False
        cache_miss_dict = {'0':-1}
        while not finished_signal and -1 in cache_miss_dict.values():
            if DEBUG:
                print(f"[Worker][RANK {self.rank}] Poll requests...")
            future_call_poll = rpc.rpc_async(to=coordinator_owner,func=call_remote_method, 
                                         args=(CacheCoordinator.poll,self.coordinator_rref,task_info_list))
            res = future_call_poll.wait()
            if DEBUG:
                print(f"[Worker][RANK {self.rank}] Poll result: {res} Task info list: {[task['id'] for task in task_info_list]}")
            finished_signal = res[0]
            cache_miss_dict = res[1]
            if DEBUG:
                print(f"[Worker][RANK {self.rank}] Cache miss dict: {cache_miss_dict}")
            if finished_signal and -1 not in cache_miss_dict.values():
                if DEBUG:
                    print(f"[Worker][RANK {self.rank}] Requests finished")
            else:
                if DEBUG:
                    print(f"[Worker][RANK {self.rank}] Requests are still being processed...")
                time.sleep(5)
            if 0 in cache_miss_dict.values():
                pass
                # print(f"[Worker][RANK {self.rank}] Cache miss detected")
            self.cache_miss_dict[req_id] = cache_miss_dict
        # print(f"[Worker][RANK {self.rank}] Moving compute buffer to device {self.gpu_index}...")
        # self.compute_buffer.to(self.device)

    def forward_with_computation(self, task_info_list:List):
        time1 = time.time()
        coordinator_owner = self.coordinator_rref.owner()
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Add {len(task_info_list)} requests to coordinator")
        rpc.rpc_sync(to=coordinator_owner, 
                         func=call_remote_method, 
                         args=(CacheCoordinator.add_requests,self.coordinator_rref, 
                               task_info_list))
        time2 = time.time()
        
        future_call_poll = rpc.rpc_async(to=coordinator_owner,func=call_remote_method, 
                            args=(CacheCoordinator.poll_batch,self.coordinator_rref,task_info_list))
        cache_miss_dict = future_call_poll.wait()
        for req_id in cache_miss_dict:
            self.cache_miss_dict[req_id] = cache_miss_dict[req_id]

        ## start model inference
        time3 = time.time()
        queried_task_info_list = process_task_info(task_info_list)
        attn_metadata, cached_tokens = prepare_attention_meta(queried_task_info_list, self.local_kv_cache_block_size, self.local_max_kv_cache_blocks, self.device)
        input_ids = torch.zeros(attn_metadata.nnz_qo, dtype=torch.int32, device=self.device)
        positions = torch.arange(attn_metadata.nnz_qo, dtype=torch.int64, device=self.device)
        time4 = time.time()
        # print(f"shape {input_ids.shape} {cached_tokens}")
        output = self.model(input_ids, positions, self.local_kvcache, attn_metadata)    
        time5 = time.time()
        print(f"worker {self.worker_index}, read kv cache time {time3-time2}s, compute time: {time5-time3}s, 100Gbps network time: {(cached_tokens*model_params['num_kv_heads']*model_params['head_size']*model_params['num_layers']*2*2)/(12*1000*1000*1000)}s")

    def receive_task_info(self, task_info_list):
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv taskinfo length:{len(task_info_list)} from scheduler")
        for task_info in task_info_list:
            req_id = task_info['request_id']
            self.buffer_control_dict[req_id] = []
            for i in task_info_list:
                next_pos = self.start_pos+i['token_num']
                if next_pos > self.buffer_size:
                    # buffer满了就从头开始
                    # 做出cache管理之前的权宜之计
                    self.start_pos = 0
                    next_pos = i['token_num']
                self.buffer_control_dict[req_id].append((self.start_pos,next_pos))
                self.start_pos = next_pos
        self.forward_with_computation(task_info_list)
        # self.preprare_send_data(task_info_list)

    def receive_kvcache_data(self, task_info):
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv kvcache data {task_info} from kvcache")
        self.write_compute_buffer(task_info)

    def send_kvcache_data(self, task_info):
        dst_rank = task_info['cache_worker'] + KVCACHE_offset
        request_id = task_info['request_id']
        data_length = task_info['data_length']
        ind = task_info['index']
        # start_pos,offest = self.buffer_control_dict[request_id][ind]
        start_pos,offest = 24,24+42
        # if DEBUG:
        print(f"[Worker][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={data_length}")
        # TODO 实际发的数据从哪里来
        dist.send(tensor=self.compute_buffer[start_pos:offest], dst=dst_rank)
        # if DEBUG:
        print(f"[Worker][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={data_length}")

    def write_compute_buffer(self, task_info):
        cache_worker = task_info['cache_worker']
        token_num = task_info['token_num']
        # req_id = task_info['request_id']
        # ind = task_info['index']
        # start_pos,offest = self.buffer_control_dict[req_id][ind] 
        # NOTE: disable buffer management temporarily
        src_rank = cache_worker + KVCACHE_offset
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Writting kvcache data from Rank {src_rank}")
        recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
        )
        dist.recv(tensor=recv_tensor, src=src_rank)
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv tensor from Rank {src_rank}: {recv_tensor}")
        # 在这里使用to(device)会导致卡死
        # self.compute_buffer[start_pos:offest] = recv_tensor

    def preprare_send_data(self,task_info_list):
        coordinator_owner = self.coordinator_rref.owner()
        request_id = task_info_list[0]['request_id']
        cache_miss_dict = self.cache_miss_dict.get(request_id,{})
        send_task_list = []
        for task_info in task_info_list:
            item_id = task_info['id']
            if cache_miss_dict.get(item_id) == 0:
                task_info['task_type'] = SIGNAL_RECV
                send_task_list.append(task_info)
        # print(f"[Worker][RANK {self.rank}] Sending data to kvcache")
        rpc.rpc_sync(to=coordinator_owner, 
                         func=call_remote_method, 
                         args=(CacheCoordinator.add_requests,self.coordinator_rref, 
                               send_task_list))