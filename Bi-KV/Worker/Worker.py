from ast import List
import time
import torch.distributed.rpc as rpc
import torch.distributed as dist
from inputGenerator.inputGenerator import InputPrompt
from rpc_def import *
from DistributedStorage.cachescoordinator import CacheCoordinator
from Remote.remote_call import call_remote_method
import torch
from config import *

class Worker:
    def __init__(self, rank, coordinator_rref):
        self.rank = rank
        self.worker_index=rank-WORKER_offset
        self.coordinator_rref = coordinator_rref
        self.gpu_index = rank
        self.device = torch.device(f"cuda:{self.gpu_index}")
        self.compute_buffer = torch.full(
            (1024 * 1024 * 10,),
            self.rank,
            device='cpu',
            dtype=torch.float16
        )
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
        self.forward(task_info_list)

    def receive_kvcache_data(self, task_info):
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv kvcache data {task_info} from kvcache")
        self.write_compute_buffer(task_info)

    def send_kvcache_data(self, task_info):
        dst_rank = task_info['recv_worker'] + KVCACHE_offset
        request_id = task_info['request_id']
        data_length = task_info['data_length']
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={data_length}")
        # TODO 实际发的数据从哪里来
        dist.send(tensor=self.compute_buffer[:data_length], dst=dst_rank)
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={data_length}")

    def write_compute_buffer(self, task_info):
        send_worker = task_info['send_worker']
        data_length = task_info['data_length']
        src_rank = send_worker + KVCACHE_offset
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Writting kvcache data from Rank {src_rank}, length: {data_length}")
        # received_tensor = torch.empty_like(self.compute_buffer)
        received_tensor = torch.empty(data_length, dtype=torch.float16)
        dist.recv(tensor=received_tensor, src=src_rank)
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv tensor from Rank {src_rank}: {received_tensor}")
        # 在这里使用to(device)会导致卡死
        self.compute_buffer = received_tensor