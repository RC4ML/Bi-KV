import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
from rpc_def import *
# worker 2,3,4.5
# cache  6,7,8,9
class KVCache:
    def __init__(self, rank):
        """初始化 KVCache"""
        self.rank = rank + KVCACHE_offset
        self.gpu_index=rank
        self.cache_data = torch.full(
            (1024 * 1024 * 10,),  # Tensor 的大小（10MB）
            self.rank,
            device=f'cuda:{rank}',
            dtype=torch.float32
        )
        print(f"[KVCache][GPU index:{rank}rank: {self.rank}] 初始化：Tensor大小={self.cache_data.size()}，值={self.rank}")

    def send_data(self, send_gpu, recv_gpu, request_id):
        """GPU发送数据"""
        print(f"[KVCache][GPU {send_gpu}] 开始发送数据到 GPU {recv_gpu}, 请求ID={request_id}")
        dst_rank=recv_gpu+KVCACHE_offset
        dist.send(tensor=self.cache_data, dst=dst_rank)  # 使用 NCCL 后端进行 GPU 之间的数据传递
        print(f"[KVCache][GPU {send_gpu}][rank{self.rank}] 完成发送数据到 GPU {recv_gpu},[rank{dst_rank}] 请求ID={request_id}")

    def receive_data(self, send_gpu, request_id):
        """GPU接收数据"""
        print(f"[KVCache][GPU {self.rank}] 开始接收数据从 GPU {send_gpu}, 请求ID={request_id}")
        received_tensor = torch.empty_like(self.cache_data)
        src_rank=send_gpu+KVCACHE_offset
        dist.recv(tensor=received_tensor, src=src_rank)  # 使用 NCCL 后端进行 GPU 之间的数据传递
        print(f"[KVCache][GPU {self.gpu_index}] [rank{self.rank}]完成接收数据从 GPU {send_gpu}[rank{src_rank}], 请求ID={request_id}, receive_data={received_tensor}")
        # self.cache_data = received_tensor

    def send_confirmation(self, confirmation_msg):
        """向调度器发送确认信息"""
        print(f"[KVCache][GPU {self.rank}] 发送确认消息到调度器: 请求ID={confirmation_msg}")
        return confirmation_msg

    def terminate(self):
        """处理终止信号"""
        print(f"[KVCache][GPU {self.rank}] 收到终止信号，退出运行")
        return "Terminated"
    
    def receive_task_info(self,task_info):
        """模拟接收任务信息的方法"""
        # 这里可以模拟任务的传入，或由调度器通过 RPC 给每个 GPU 发送任务信息
        print(f"[KVCache][RANK {self.rank}] taskinfo is{task_info}")
        task_type, request_id, send_gpu, recv_gpu = task_info
        if task_type == SIGNAL_SEND:  # 发送数据
            self.send_data(send_gpu, recv_gpu, request_id)
        elif task_type == SIGNAL_RECV:  # 接收数据
            self.receive_data(send_gpu, request_id)
            return request_id           # 接收成功返回ACK