from ast import List
import time
import torch.distributed.rpc as rpc
import torch.distributed as dist
from inputGenerator.inputGenerator import InputPrompt
from rpc_def import *
from DistributedStorage.CacheCoordinator import CacheCoordinator
from DistributedStorage.Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
from Remote.remote_call import call_remote_method
import torch
from config import *

# TODO 提取model_params作为公共参数
model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

token_shape = (model_params['head_size'],
               model_params['num_kv_heads'],
               model_params['num_layers'])

class Worker:
    def __init__(self, rank, coordinator_rref):
        self.rank = rank
        self.worker_index=rank-WORKER_offset
        self.coordinator_rref = coordinator_rref
        self.gpu_index = rank
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


    def receive_task_info(self, task_info_list):
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv taskinfo length:{len(task_info_list)} from scheduler")
        req_id = task_info_list[0]['request_id']
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
        self.forward(task_info_list)
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
        data_length = task_info['data_length']
        token_num = task_info['token_num']
        req_id = task_info['request_id']
        ind = task_info['index']
        start_pos,offest = self.buffer_control_dict[req_id][ind]
        src_rank = cache_worker + KVCACHE_offset
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Writting kvcache data from Rank {src_rank}, length: {data_length}")
        # received_tensor = torch.empty_like(self.compute_buffer)
        recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
        )
        dist.recv(tensor=recv_tensor, src=src_rank)
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv tensor from Rank {src_rank}: {recv_tensor}")
        # 在这里使用to(device)会导致卡死
        self.compute_buffer[start_pos:offest] = recv_tensor

    def preprare_send_data(self,task_info_list):
        coordinator_owner = self.coordinator_rref.owner()
        request_id = task_info_list[0]['request_id']
        cache_miss_dict = self.cache_miss_dict.get(request_id,{})
        # print(f"[Worker][RANK {self.rank}] Preparing... {self.cache_miss_dict} {cache_miss_dict} {request_id}")
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