import torch.distributed.rpc as rpc
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
        print(f"[Worker] init Worker rank {self.rank}")

    def forward(self, task_info):
        coordinator_owner = self.coordinator_rref.owner()
        print(f"Worker {self.worker_index} add request{task_info} to coordinator")
        request_id, send_cpu, recv_cpu = task_info
        future_call_coordin = rpc.rpc_async(to=coordinator_owner, func=_call_remote_method, args=(CacheCoordinator.add_request,self.coordinator_rref, request_id, send_cpu, recv_cpu))
        future_call_coordin.wait()

    def receive_task_info(self, task_info):
        print(f"[Worker][RANK {self.rank}] recv taskinfo {task_info} from scheduler")
        self.forward(task_info)
