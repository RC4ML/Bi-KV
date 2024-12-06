from turtle import forward
import torch.distributed.rpc as rpc

from inputGenerator.inputGenerator import InputPrompt

# Worker 类
class Worker:
    def __init__(self, rank):
        self.rank = rank

    def regester_worker(self):
        # TODO dynamic world size
        pass

    def forward(self,prompt:InputPrompt):
        # TODO Add communication with cache
        pass

    def receive_task_info(self,task_info:InputPrompt):
        """模拟接收任务信息的方法"""
        # 这里可以模拟任务的传入，或由调度器通过 RPC 给每个 GPU 发送任务信息
        print(f"[Worker][RANK {self.rank}] taskinfo is {task_info.user_id}")
        self.forward(task_info)