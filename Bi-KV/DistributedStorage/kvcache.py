from ast import Dict
import random

from numpy import rec
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from DistributedStorage.Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
from rpc_def import KVCACHE_offset,WORKER_offset
from Remote.remote_call import call_remote_method
from config import *

# TODO 提取model_params作为公共参数
model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

token_shape = (model_params['head_size'],
               model_params['num_kv_heads'],
               model_params['num_layers'])

class KVCache:
    def __init__(self, rank):
        self.rank = rank + KVCACHE_offset
        self.cpu_index = rank
        self.cache_size = 1000
        self.cache_control_dict = {}
        self.cache_data = torch.full(
            (self.cache_size,) + token_shape, 
            self.rank,
            device='cpu',
            dtype=torch.float16
        )
        self.start_pos = 0
        print(f"[KVCache][CPU index:{rank} rank: {self.rank}] 初始化：Tensor大小={self.cache_data.size()}，值={self.rank}")

    def send_data(self,task_info:Dict):
        dst_rank = task_info['recv_worker'] + WORKER_offset
        request_id = task_info['request_id']
        token_num = task_info['token_num']
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={token_num}")
        # TODO 实际的读写逻辑大概不是这样
        start_pos = random.randint(0,self.cache_size/2)
        send_tensor = self.cache_data[start_pos:start_pos+token_num]
        dist.send(tensor=send_tensor, dst=dst_rank)
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={token_num}")

    def receive_data(self, task_info:Dict):
        request_id = task_info['request_id']
        send_worker = task_info['recv_worker']
        token_num = task_info['token_num']
        item_id = task_info['id']
        src_rank = send_worker + WORKER_offset
        # if DEBUG:
        print(f"[KVCache][Rank {self.rank}] 开始接收数据从 Rank {src_rank}, 请求ID={request_id}")
        recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
        )
        dist.recv(tensor=recv_tensor, src=src_rank)
        # if DEBUG:
        print(f"[KVCache][CPU {self.cpu_index}] [rank{self.rank}] 完成接收数据从 Rank {send_worker} [rank{src_rank}], 请求ID={request_id}")
        next_pos = self.start_pos + token_num
        if next_pos > self.cache_size:
            self.start_pos = 0
            next_pos = self.start_pos + token_num
        # self.cache_data[self.start_pos:next_pos] = recv_tensor
        self.cache_control_dict[item_id] = (self.start_pos,next_pos)
        self.start_pos = next_pos

    def send_confirmation(self, confirmation_msg):
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 发送确认消息到调度器: 请求ID={confirmation_msg}")
        return confirmation_msg

    def terminate(self):
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 收到终止信号，退出运行")
        return "Terminated"
    
    def receive_task_info(self, task_info:Dict, worker_ref):
        from Worker.Worker import Worker
        if DEBUG:
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
            recv_worker = task_info['recv_worker']
            print(f"[KVCache] 执行请求 {request_id} - Rank {recv_worker+WORKER_offset} -> Rank {send_worker+KVCACHE_offset}")
            remote_send = rpc.rpc_async(
                to=worker_ref.owner(), func=call_remote_method, 
                args=(Worker.send_kvcache_data,worker_ref, task_info))
            self.receive_data(task_info)
            remote_send.wait()
            return request_id
