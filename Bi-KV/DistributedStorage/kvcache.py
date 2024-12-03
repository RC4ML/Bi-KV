import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE

class KVCache:
    def __init__(self, rank):
        """初始化 KVCache"""
        self.rank = rank
        torch.cuda.set_device(rank -1)
        print(f"kv cache被初始化在GPU{torch.cuda.current_device()}上")

    def terminate(self):
        """处理终止信号"""
        print(f"[KVCache][GPU {self.rank}] 收到终止信号，退出运行")
        dist.destroy_process_group()
    def receive_task_info(self,task_info):
        """模拟接收任务信息的方法"""
        # 这里可以模拟任务的传入，或由调度器通过 RPC 给每个 GPU 发送任务信息
        print(f"[KVCache][RANK {self.rank}] taskinfo is{task_info}")
        dist.barrier()  # 等待所有进程完成
        dist.destroy_process_group()