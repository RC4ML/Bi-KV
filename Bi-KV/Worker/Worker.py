from ast import List
import torch.distributed.rpc as rpc
import torch.distributed as dist
from inputGenerator.inputGenerator import InputPrompt
from rpc_def import *
from DistributedStorage.cachescoordinator import CacheCoordinator
from Remote.remote_call import call_remote_method
import torch


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
        print(f"[Worker][RANK {self.rank}] Add {len(task_info_list)} requests to coordinator")
        rpc.rpc_sync(to=coordinator_owner, 
                         func=call_remote_method, 
                         args=(CacheCoordinator.add_requests,self.coordinator_rref, 
                               task_info_list))
        # for task_info in task_info_list:
        #     print(f"[Worker][RANK {self.rank}] Add request{task_info} to coordinator")
        #     request_id, recv_worker = task_info['request_id'], task_info['recv_worker']
        #     rpc.rpc_sync(to=coordinator_owner, 
        #                  func=call_remote_method, 
        #                  args=(CacheCoordinator.add_request,self.coordinator_rref, 
        #                        task_info))
        print(f"[Worker][RANK {self.rank}] Poll requests...")
        # future_call_poll = rpc.rpc_async(to=coordinator_owner,func=call_remote_method, 
        #                                  args=(CacheCoordinator.process_requests,self.coordinator_rref))
        future_call_poll = rpc.rpc_async(to=coordinator_owner,func=call_remote_method, 
                                         args=(CacheCoordinator.poll,self.coordinator_rref,task_info_list))
        res = future_call_poll.wait()
        # TODO 更细致地处理poll结果
        if res:
            print(f"[Worker][RANK {self.rank}] Requests finished")
        else:
            print(f"[Worker][RANK {self.rank}] Requests are still being processed...")
        # print(f"[Worker][RANK {self.rank}] Moving compute buffer to device {self.gpu_index}...")
        # self.compute_buffer.to(self.device)


    def receive_task_info(self, task_info_list):
        print(f"[Worker][RANK {self.rank}] Recv taskinfo length:{len(task_info_list)} from scheduler")
        self.forward(task_info_list)

    def receive_kvcache_data(self, task_info):
        print(f"[Worker][RANK {self.rank}] Recv kvcache data {task_info} from kvcache")
        self.write_compute_buffer(task_info)

    def write_compute_buffer(self, task_info):
        send_worker = task_info['send_worker']
        data_length = task_info['data_length']
        src_rank = send_worker + KVCACHE_offset
        print(f"[Worker][RANK {self.rank}] Writting kvcache data from Rank {src_rank}, length: {data_length}")
        # received_tensor = torch.empty_like(self.compute_buffer)
        received_tensor = torch.empty(data_length, dtype=torch.float16)
        dist.recv(tensor=received_tensor, src=src_rank)
        print(f"[Worker][RANK {self.rank}] Recv tensor from Rank {src_rank}: {received_tensor}")
        # 在这里使用to(device)会导致卡死
        self.compute_buffer = received_tensor