"LLMScheduler.py"
from typing import List
import random
import torch.distributed.rpc as rpc

from inputGenerator.inputGenerator import InputPrompt
from Worker.Worker import Worker
from rpc_def import *
from DistributedStorage.cachescheduler import CacheScheduler
import time

# Qwen2 1.5B
model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}
PROCESS_TYPES = [
    ('scheduler', 1),
    ('coordinator', 1),
    ('inferworker', 4),
    ('kvcache', 4),
]
def get_process_info(rank, process_types=PROCESS_TYPES):
    """
    根据全局 rank 返回进程类型和该类型下的索引。

    Args:
        rank (int): 全局 rank。
        process_types (list): 进程类型及其数量的有序列表。

    Returns:
        tuple: (process_type, type_index)
    """
    current_rank = 0
    for process_type, count in process_types:
        if current_rank + count > rank:
            type_index = rank - current_rank
            return process_type, type_index
        current_rank += count
    raise ValueError(f"Rank {rank} 超出定义的进程类型范围。")
def _call_Worker_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)

class LLMScheduler:
    def __init__(self, world_size:int):
        # self.worker_func = worker_func
        self.world_size = world_size
        self.num_workers = WORKER_NUM
        self.strategy_mode = "Default"
        self.coordinator_ref = []
        self.worker_ref = []
        self.kvcache_ref = []
        self.prompt_list = []
        for r in range(1, 6):
            proc_type, proc_index = get_process_info(r)
            rpc_info = rpc.get_worker_info(f"{proc_type}{proc_index}")
            if r == 1:  # 初始化 coordinator 和 kvcache
                self.coordinator_ref.append(
                    rpc.remote(to=rpc_info, func=CacheScheduler, args=(r, KVCACHE_NUM))
                )
                print("[LLMScheduler]finish initcoordinator kvcache ")
            else:          # 初始化 worker
                self.worker_ref.append(
                    rpc.remote(to=rpc_info, func=Worker, args=(r, self.coordinator_ref[0]))
                )
        time.sleep(1)
        print("[LLMScheduler] finish init all class")

    def add_prompt_list(self, prompt_list):
        self.prompt_list = prompt_list

    def process_prompt(self):
        futures = []
        for prompt in self.prompt_list:
            future = self._send_prompt(prompt)
            futures.append(future)
        
    def _send_prompt(self, prompt):
        # 发送请求到相应的 worker
        request_id, send_cpu, recv_cpu = prompt
        worker_id = recv_cpu  # 这里使用接收方CPU和worker在同一设备上
        send_worker_ref = self.worker_ref[worker_id]
        owner_worker_ref = send_worker_ref.owner()  # 获取 RRef 的拥有者
        future = rpc.rpc_sync(to=owner_worker_ref, func=_call_Worker_method, args=(Worker.receive_task_info, send_worker_ref, prompt))
        return future
    
    def strategy(self, user_id: int) -> int:
        # schedule strategy
        if self.strategy_mode == "WIP":
            return random.randint(self.num_workers)
        else:
            return user_id % self.num_workers

    def shutdown(self):
        rpc.shutdown()

def PromptOrder(prompt: InputPrompt) -> str:
    user_tokens = prompt.user_history_tokens
    item_tokens = sum([item.token_count for item in prompt.items])

    if user_tokens > item_tokens:
        return "User History First"
    else:
        return "Item First"
