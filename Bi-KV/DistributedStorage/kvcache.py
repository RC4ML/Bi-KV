from ast import Dict
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from DistributedStorage.Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
from rpc_def import KVCACHE_offset,WORKER_offset
from Remote.remote_call import call_remote_method

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

    def send_data(self,task_info:Dict):
        dst_rank = task_info['recv_worker'] + WORKER_offset
        request_id = task_info['request_id']
        token_num = task_info['token_num']
        print(f"[KVCache][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 请求ID={request_id}")
        # TODO 实际的读写逻辑大概不是这样
        dist.send(tensor=self.cache_data[:token_num], dst=dst_rank)
        print(f"[KVCache][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 请求ID={request_id}")

    def receive_data(self, send_cpu, request_id):
        src_rank = send_cpu + KVCACHE_offset
        print(f"[KVCache][Rank {self.rank}] 开始接收数据从 Rank {src_rank}, 请求ID={request_id}")
        received_tensor = torch.empty_like(self.cache_data)
        dist.recv(tensor=received_tensor, src=src_rank)
        print(f"[KVCache][CPU {self.cpu_index}] [rank{self.rank}] 完成接收数据从 Rank {send_cpu} [rank{src_rank}], 请求ID={request_id}, receive_data={received_tensor}")

    def send_confirmation(self, confirmation_msg):
        print(f"[KVCache][Rank {self.rank}] 发送确认消息到调度器: 请求ID={confirmation_msg}")
        return confirmation_msg

    def terminate(self):
        print(f"[KVCache][Rank {self.rank}] 收到终止信号，退出运行")
        return "Terminated"
    
    def receive_task_info(self, task_info:Dict, worker_ref):
        from Worker.Worker import Worker
        print(f"[KVCache][RANK {self.rank}] taskinfo is {task_info}")
        # task_type, request_id, send_worker, recv_worker = task_info
        task_type = task_info['task_type']
        request_id = task_info['request_id']
        if task_type == SIGNAL_SEND:
            remote_recv = rpc.rpc_async(
                to=worker_ref.owner(), func=call_remote_method, 
                args=(Worker.receive_kvcache_data,worker_ref, task_info))
            self.send_data(task_info)
            remote_recv.wait()
            return request_id
        elif task_type == SIGNAL_RECV:
            send_worker = task_info['send_worker']
            self.receive_data(send_worker, request_id)
            return request_id
