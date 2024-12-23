from ast import List
import time
import torch.distributed.rpc as rpc
import torch.distributed as dist
from inputGenerator.inputGenerator import InputPrompt
from rpc_def import *
from DistributedStorage.CacheCoordinator import CacheCoordinator
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
        print(f"[Worker][RANK {self.rank}] Init Worker")

    def forward(self, task_info_list:List):
        coordinator_owner = self.coordinator_rref.owner()
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Add {len(task_info_list)} requests to coordinator")
        rpc.rpc_sync(to=coordinator_owner, 
                         func=call_remote_method, 
                         args=(CacheCoordinator.add_requests,self.coordinator_rref, 
                               task_info_list))
        res = False
        while not res:
            if DEBUG:
                print(f"[Worker][RANK {self.rank}] Poll requests...")
            future_call_poll = rpc.rpc_async(to=coordinator_owner,func=call_remote_method, 
                                         args=(CacheCoordinator.poll,self.coordinator_rref,task_info_list))
            res = future_call_poll.wait()
            if res:
                if DEBUG:
                    print(f"[Worker][RANK {self.rank}] Requests finished")
            else:
                if DEBUG:
                    print(f"[Worker][RANK {self.rank}] Requests are still being processed...")
                time.sleep(5)
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

    def receive_kvcache_data(self, task_info):
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv kvcache data {task_info} from kvcache")
        self.write_compute_buffer(task_info)

    def send_kvcache_data(self, task_info):
        dst_rank = task_info['recv_worker'] + KVCACHE_offset
        request_id = task_info['request_id']
        data_length = task_info['data_length']
        ind = task_info['index']
        # start_pos,offest = self.buffer_control_dict[request_id][ind]
        start_pos,offest = 24,24+42
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={data_length}")
        # TODO 实际发的数据从哪里来
        dist.send(tensor=self.compute_buffer[start_pos:offest], dst=dst_rank)
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={data_length}")

    def write_compute_buffer(self, task_info):
        send_worker = task_info['send_worker']
        data_length = task_info['data_length']
        token_num = task_info['token_num']
        req_id = task_info['request_id']
        ind = task_info['index']
        start_pos,offest = self.buffer_control_dict[req_id][ind]
        src_rank = send_worker + KVCACHE_offset
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