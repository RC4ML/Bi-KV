import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from DistributedStorage.Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
from rpc_def import KVCACHE_offset

class KVCache:
    def __init__(self, rank):
        self.rank = rank + KVCACHE_offset
        self.cpu_index = rank
        self.cache_data = torch.full(
            (1024 * 1024 * 10,),
            self.rank,
            device='cpu',
            dtype=torch.float32
        )
        print(f"[KVCache][CPU index:{rank} rank: {self.rank}] 初始化：Tensor大小={self.cache_data.size()}，值={self.rank}")

    def send_data(self, send_cpu, recv_cpu, request_id):
        print(f"[KVCache][CPU {send_cpu}] 开始发送数据到 CPU {recv_cpu}, 请求ID={request_id}")
        dst_rank = recv_cpu + KVCACHE_offset
        dist.send(tensor=self.cache_data, dst=dst_rank)
        print(f"[KVCache][CPU {send_cpu}] 完成发送数据到 CPU {recv_cpu}, [rank{dst_rank}] 请求ID={request_id}")

    def receive_data(self, send_cpu, request_id):
        print(f"[KVCache][CPU {self.rank}] 开始接收数据从 CPU {send_cpu}, 请求ID={request_id}")
        received_tensor = torch.empty_like(self.cache_data)
        src_rank = send_cpu + KVCACHE_offset
        dist.recv(tensor=received_tensor, src=src_rank)
        print(f"[KVCache][CPU {self.cpu_index}] [rank{self.rank}] 完成接收数据从 CPU {send_cpu} [rank{src_rank}], 请求ID={request_id}, receive_data={received_tensor}")

    def send_confirmation(self, confirmation_msg):
        print(f"[KVCache][CPU {self.rank}] 发送确认消息到调度器: 请求ID={confirmation_msg}")
        return confirmation_msg

    def terminate(self):
        print(f"[KVCache][CPU {self.rank}] 收到终止信号，退出运行")
        return "Terminated"
    
    def receive_task_info(self, task_info):
        print(f"[KVCache][RANK {self.rank}] taskinfo is {task_info}")
        task_type, request_id, send_cpu, recv_cpu = task_info
        if task_type == SIGNAL_SEND:
            self.send_data(send_cpu, recv_cpu, request_id)
        elif task_type == SIGNAL_RECV:
            self.receive_data(send_cpu, request_id)
            return request_id
