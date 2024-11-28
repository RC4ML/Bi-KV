import torch
import torch.distributed as dist

class KVCache:
    def __init__(self, rank):
        """初始化 KVCache"""
        self.rank = rank
        self.cache_data = torch.full(
            ((1024 * 1024 * 10),),  # 确保 size 是一个元组
            self.rank, 
            device=torch.device('cuda', self.rank), 
            dtype=torch.float32
        )
        print(f"[KVCache][GPU {self.rank}] 初始化：Tensor大小={self.cache_data.size()}，值={self.rank}")

    def send_data(self, send_gpu, recv_gpu, request_id):
        """发送数据"""
        print(f"[KVCache][GPU {send_gpu}] 开始发送数据到 GPU {recv_gpu}, 请求ID={request_id}")
        dist.send(tensor=self.cache_data, dst=recv_gpu)
        print(f"[KVCache][GPU {send_gpu}] 完成发送数据到 GPU {recv_gpu}, 请求ID={request_id}")

    def receive_data(self, send_gpu, recv_gpu, request_id):
        """接收数据"""
        print(f"[KVCache][GPU {recv_gpu}] 开始接收数据从 GPU {send_gpu}, 请求ID={request_id}")
        received_tensor = torch.empty_like(self.cache_data)
        dist.recv(tensor=received_tensor, src=send_gpu)
        print(f"[KVCache][GPU {recv_gpu}] 完成接收数据从 GPU {send_gpu}, 请求ID={request_id}")
        self.cache_data = received_tensor
        confirmation_tensor = torch.tensor([request_id], device=self.cache_data.device, dtype=torch.int32)
        dist.send(tensor=confirmation_tensor, dst=0)  # 发送确认消息到调度器
        print(f"[KVCache][GPU {recv_gpu}] 发送确认消息到调度器，请求ID={request_id}")

    def run(self):
        """监听请求"""
        print(f"[KVCache][GPU {self.rank}] 开始监听请求")
        while True:
            task_info = torch.zeros(4, device=self.cache_data.device, dtype=torch.int32)
            dist.recv(tensor=task_info, src=0)
            print(f"[KVCache][GPU {self.rank}] 接收到的请求", task_info)
            task_type, request_id, send_gpu, recv_gpu = task_info[0].item(), task_info[1].item(), task_info[2].item(), task_info[3].item()

            if task_type == 0:  # 发送数据
                self.send_data(send_gpu, recv_gpu, request_id)
            elif task_type == 1:  # 接收数据
                self.receive_data(send_gpu, recv_gpu, request_id)