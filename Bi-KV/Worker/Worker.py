from ast import Dict, List
import time

import torch.distributed.rpc as rpc
import torch.distributed as dist
from rpc_def import *
from DistributedStorage.CacheCoordinator import CacheCoordinator
from DistributedStorage.PageManager import PageManager
from DistributedStorage.Signals import SIGNAL_RECV,CACHE_MISS
from Remote.remote_call import call_remote_method
import torch
from config import *
from Model.qwen2 import process_task_info, prepare_attention_meta, Qwen2ForCausalLM, AttentionMetadata, token_shape, model_params
from transformers import Qwen2Config
from vllm.config import CacheConfig

import time

class Worker:
    def __init__(self, rank, coordinator_rref):
        self.rank = rank
        self.worker_index=rank-WORKER_offset
        self.coordinator_rref = coordinator_rref
        self.gpu_index = rank # NOTE: set to rank in a single machine
        self.device = torch.device(f"cuda:{self.gpu_index}")
        # key item id value(start_pos,offset) req_id?
        # 多个req_id并发的情况？
        self.buffer_size = 10000
        self.page_size = 50
        # PageManager会不会遇到并发？？？
        self.page_manager = PageManager(cache_size=self.buffer_size, page_size=self.page_size)
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
        while not finished_signal and (-1 in cache_miss_dict.values() or cache_miss_dict == {}):
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
        # 这里的task_info同样有着一样的req_id
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
        # cache_miss_dict是一个嵌套字典，第一层是req_id，第二层是item_id
        cache_miss_dict = future_call_poll.wait()
        for req_id in cache_miss_dict:
            self.cache_miss_dict[req_id] = cache_miss_dict[req_id]

        # ## start model inference
        # time3 = time.time()
        # queried_task_info_list = process_task_info(task_info_list)
        # attn_metadata, cached_tokens = prepare_attention_meta(queried_task_info_list, self.local_kv_cache_block_size, self.local_max_kv_cache_blocks, self.device)
        # input_ids = torch.zeros(attn_metadata.nnz_qo, dtype=torch.int32, device=self.device)
        # positions = torch.arange(attn_metadata.nnz_qo, dtype=torch.int64, device=self.device)
        # time4 = time.time()
        # # print(f"shape {input_ids.shape} {cached_tokens}")
        # output = self.model(input_ids, positions, self.local_kvcache, attn_metadata)    
        # time5 = time.time()
        # print(f"worker {self.worker_index}, read kv cache time {time3-time2}s, compute time: {time5-time3}s, 100Gbps network time: {(cached_tokens*model_params['num_kv_heads']*model_params['head_size']*model_params['num_layers']*2*2)/(12*1000*1000*1000)}s")

    def receive_task_info(self, task_info_list):
        # 此时的task_info_list是一个request的所有task，req_id相同
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv taskinfo length:{len(task_info_list)} from scheduler")
        for task_info in task_info_list:
            item_id = task_info['id']
            if item_id == -1:
                continue
            self.page_manager.load_item(item_id, task_info['token_num'])
            self.page_manager.set_protected(item_id)
        self.forward_with_computation(task_info_list)
        # print(f"[Worker][RANK {self.rank}] Sending data to kvcache")
        self.preprare_send_data(task_info_list)

    def receive_task_info_batch(self, task_info_list):
        # 按照req_id分组
        task_info_dict = {}
        for i in task_info_list:
            req_id = i['request_id']
            if req_id not in task_info_dict:
                task_info_dict[req_id] = []
            task_info_dict[req_id].append(i)
        for i in task_info_dict.values():
            self.receive_task_info(i)

    def receive_kvcache_data(self, task_info):
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv kvcache data {task_info} from kvcache")
        self.write_compute_buffer(task_info)

    def receive_kvcache_data_batch(self, combined_task_info):
        cache_worker = combined_task_info['cache_worker']
        src_rank = cache_worker + KVCACHE_offset
        token_num = combined_task_info['token_num']
        recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
        )
        dist.recv(tensor=recv_tensor, src=src_rank)
        # TODO 写入到buffer中
        start_pos = 0
        for id_pair in combined_task_info['id_token_pair']:
            item_id = id_pair[0]
            offset = id_pair[1]
            item_tensor = recv_tensor[start_pos:start_pos+offset]
            start_pos += offset
            # 让page manager管理buffer
            if item_id not in self.page_manager.get_loaded_lists():
                page_set, _ = self.page_manager.load_item(item_id, offset)
            else:
                page_set = self.page_manager.access_item(item_id)
            # 按照得到的page写入buffer
            for idx,page in enumerate(page_set):
                if idx == len(page_set) - 1:
                    self.compute_buffer[page*self.page_size:page*self.page_size + offset % self.page_size] \
                        = item_tensor[idx*self.page_size:]
                else:
                    self.compute_buffer[page*self.page_size:(page+1)*self.page_size] = item_tensor[idx*self.page_size:(idx+1)*self.page_size]

    def send_kvcache_data(self, task_info):
        dst_rank = task_info['cache_worker'] + KVCACHE_offset
        token_num = task_info['token_num']
        id = task_info['id']
        # if id not in self.buffer_control_dict:
        #     print(f"[Worker][RANK {self.rank}] Error: id {id} not in buffer control dict")
        start_pos,offest = self._manage_buffer(id, token_num)
        # if DEBUG:
        print(f"[Worker][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 长度={token_num}")
        dist.send(tensor=self.compute_buffer[start_pos:offest], dst=dst_rank)
        # if DEBUG:
        print(f"[Worker][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 长度={token_num}")

    def send_kvcache_data_batch(self, combined_task_info):
        dst_rank = combined_task_info['cache_worker'] + KVCACHE_offset
        token_num = combined_task_info['token_num']
        id_pair_list = combined_task_info['id_token_pair']
        id_list = [i[0] for i in id_pair_list]
        send_tensor_list = []
        for id_pair in id_pair_list:
            i = id_pair[0]
            token_num = id_pair[1]
            if i not in self.page_manager.get_loaded_lists():
                print(f"[Worker][RANK {self.rank}][{time.time()}] Error: id {i} not in page manager")
            page_set = self.page_manager.access_item(i)
            send_tensor = torch.empty(
                (token_num,) + token_shape, 
                dtype=torch.float16
            )
            for idx,page in enumerate(page_set):
                if idx == len(page_set) - 1:
                    send_tensor[idx*self.page_size:] \
                        = self.compute_buffer[page*self.page_size:page*self.page_size+token_num%self.page_size]
                else:
                    send_tensor[idx*self.page_size:(idx+1)*self.page_size] = self.compute_buffer[page*self.page_size:(page+1)*self.page_size]
            send_tensor_list.append(send_tensor)
            self.page_manager.remove_protected(i)
        send_tensor = torch.cat(send_tensor_list, dim=0)
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 长度={token_num}")
        dist.send(tensor=send_tensor, dst=dst_rank)
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 长度={token_num}")
            

    def write_compute_buffer(self, task_info:Dict):
        cache_worker = task_info['cache_worker']
        token_num = task_info['token_num']
        # req_id = task_info['request_id']
        if task_info.get('id_token_pair') is not None:
            for i in task_info['id_token_pair']:
                self.buffer_control_dict[i] = []
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
        # 这里的task_info同样有着一样的req_id
        coordinator_owner = self.coordinator_rref.owner()
        request_id = task_info_list[0]['request_id']
        cache_miss_dict = self.cache_miss_dict.get(request_id,{})
        send_task_list = []
        for task_info in task_info_list:
            item_id = task_info['id']
            # 这里能保证item_id在cache_miss_dict中吗？
            if cache_miss_dict.get(item_id) == CACHE_MISS:
                # print(f"[Worker][RANK {self.rank}] Cache miss detected")
                task_info['task_type'] = SIGNAL_RECV
                send_task_list.append(task_info)
            else:
                # 为什么会提前解除保护？
                if item_id in self.page_manager.get_loaded_lists():
                    self.page_manager.remove_protected(item_id)
        hit_rate = sum(cache_miss_dict.values())/len(cache_miss_dict)
        if hit_rate > 0.7:
            print(f"[Worker][RANK {self.rank}] Request {request_id} Hit rate: {hit_rate} Sending {len(send_task_list)} tasks to kvcache") 
        # print(f"[Worker][RANK {self.rank}] Sending data to kvcache")
        rpc.rpc_sync(to=coordinator_owner, 
                         func=call_remote_method, 
                         args=(CacheCoordinator.add_requests,self.coordinator_rref, 
                               send_task_list))
        # 发buffer上的数据可能会被写掉？加锁？ 保证worker上的buffer没有被覆盖