from turtle import forward
import torch.distributed.rpc as rpc

from inputGenerator.inputGenerator import InputPrompt
from rpc_def import *
import torch
# Worker 类
def _call_coordinator(rref, task_info):
    print(f"[_call_coordinator]Worker add request to coordinator")
    return rref.rpc_sync().add_request(task_info)
class Worker:
    def __init__(self, rank,coordinator_rref):
        self.rank = rank
        self.coordinator_rref=coordinator_rref
        self.gpu_index=rank-WORKER_offset
        device = torch.device(f'cuda:{self.gpu_index}')
        torch.cuda.set_device(device)
        print(f"[Worker]init Worker rank{self.rank},GPU index{self.gpu_index}")

    def regester_worker(self):
        # TODO dynamic world size
        pass

    def forward(self,task_info):
        # TODO Add communication with cache
        coordinator_owner=self.coordinator_rref[0].owner()
        future_call_coordin = rpc.rpc_async(coordinator_owner, _call_coordinator,  args=(self.coordinator_rref[0], task_info))
        future_call_coordin.wait()

    def receive_task_info(self,task_info):
        """模拟接收任务信息的方法"""
        print(f"[Worker][RANK {self.rank}] taskinfo is {task_info}")
        self.forward(task_info) 