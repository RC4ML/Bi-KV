import torch.distributed.rpc as rpc
import torch.distributed as dist
from inputGenerator.inputGenerator import InputPrompt
from rpc_def import *
from DistributedStorage.cachescoordinator import CacheCoordinator
from Remote.remote_call import _call_remote_method
import torch


class Worker:
    def __init__(self, rank, coordinator_rref):
        self.rank = rank
        self.worker_index=rank-WORKER_offset
        self.coordinator_rref = coordinator_rref
        self.compute_buffer = torch.full(
            (1024 * 1024 * 10,),
            self.rank,
            device='cpu',
            dtype=torch.float32
        )
        print(f"[Worker][RANK {self.rank}] Init Worker")

    def forward(self, task_info):
        coordinator_owner = self.coordinator_rref.owner()
        print(f"[Worker][RANK {self.rank}] Add request{task_info} to coordinator")
        request_id, send_cpu, recv_cpu = task_info
        # TODO 拆分 add_request 和 poll
        future_call_coordin = rpc.rpc_async(to=coordinator_owner, func=_call_remote_method, args=(CacheCoordinator.add_request,self.coordinator_rref, request_id, send_cpu, recv_cpu))
        future_call_coordin.wait()

    def receive_task_info(self, task_info):
        print(f"[Worker][RANK {self.rank}] Recv taskinfo {task_info} from scheduler")
        self.forward(task_info)

    def receive_kvcache_data(self, task_info):
        print(f"[Worker][RANK {self.rank}] Recv kvcache data {task_info} from kvcache")
        self.write_compute_buffer(task_info)

    def write_compute_buffer(self, task_info):
        task_type, request_id, send_cpu, recv_cpu = task_info
        src_rank = send_cpu + KVCACHE_offset
        print(f"[Worker][RANK {self.rank}] Writting kvcache data from Rank {src_rank}")
        received_tensor = torch.empty_like(self.compute_buffer)
        dist.recv(tensor=received_tensor, src=src_rank)
        print(f"[Worker][RANK {self.rank}] Recv tensor: {received_tensor}")
        self.compute_buffer = received_tensor