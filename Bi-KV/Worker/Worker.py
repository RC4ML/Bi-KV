import torch.distributed.rpc as rpc
from inputGenerator.inputGenerator import InputPrompt
from rpc_def import *
import torch

def _call_coordinator(rref, task_info):
    print(f"[_call_coordinator]Worker add request to coordinator")
    request_id, send_cpu, recv_cpu = task_info
    return rref.rpc_sync().add_request(request_id, send_cpu, recv_cpu)

class Worker:
    def __init__(self, rank, coordinator_rref):
        self.rank = rank
        self.coordinator_rref = coordinator_rref
        print(f"[Worker] init Worker rank {self.rank}")

    def forward(self, task_info):
        coordinator_owner = self.coordinator_rref.owner()
        future_call_coordin = rpc.rpc_async(coordinator_owner, _call_coordinator, args=(self.coordinator_rref, task_info))
        future_call_coordin.wait()

    def receive_task_info(self, task_info):
        print(f"[Worker][RANK {self.rank}] taskinfo is {task_info}")
        self.forward(task_info)
