from typing import List
import random
import torch.distributed.rpc as rpc
from inputGenerator.inputGenerator import InputPrompt
from Worker.Worker import Worker
from rpc_def import PROCESS_TYPES, WORKER_NUM, KVCACHE_NUM, get_process_info, KVCACHE_offset
from DistributedStorage.cachescoordinator import CacheCoordinator
from Remote.remote_call import _call_remote_method
import time

model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

class LLMScheduler:
    def __init__(self, world_size:int):
        self.world_size = world_size
        self.num_workers = WORKER_NUM
        self.coordinator_ref = []
        self.worker_ref = []
        self.kvcache_ref = []
        self.prompt_list = []
        
        # 获取coordinator的rpc info
        # 根据PROCESS_TYPES: scheduler=0, coordinator=1, workers=2...(2+WORKER_NUM-1), kvcache后面
        for r in range(1, 1+1):  # coordinator只有1个，所以r=1
            proc_type, proc_index = get_process_info(r, PROCESS_TYPES)
            rpc_info = rpc.get_worker_info(f"{proc_type}{proc_index}")
            # 创建CacheCoordinator实例，传入KVCACHE_NUM
            self.coordinator_ref.append(
                rpc.remote(to=rpc_info, func=CacheCoordinator, args=(r, KVCACHE_NUM))
            )
            print("[LLMScheduler]finish init coordinator")

        # 初始化worker
        start_worker_rank = 2
        end_worker_rank = 2 + WORKER_NUM
        for r in range(start_worker_rank, end_worker_rank):
            proc_type, proc_index = get_process_info(r, PROCESS_TYPES)
            rpc_info = rpc.get_worker_info(f"{proc_type}{proc_index}")
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
        request_id, send_cpu, recv_cpu = prompt
        worker_id = recv_cpu  # 使用recv_cpu作为worker索引
        send_worker_ref = self.worker_ref[worker_id]
        owner_worker_ref = send_worker_ref.owner()  
        future = rpc.rpc_sync(to=owner_worker_ref, func=_call_remote_method, args=(Worker.receive_task_info, send_worker_ref, prompt))
        return future

    def strategy(self, user_id: int) -> int:
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
