from typing import List
import random
import torch.distributed.rpc as rpc
from inputGenerator.inputGenerator import LLMInput,InputPrompt
from Worker.Worker import Worker
from rpc_def import PROCESS_TYPES, WORKER_NUM, KVCACHE_NUM, get_process_info, KVCACHE_offset
from DistributedStorage.cachescoordinator import CacheCoordinator
from Remote.remote_call import call_remote_method
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
        self.prompt_generator:LLMInput = None
        
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
        # 将workers_rref同步到coordinator
        for r in range(1, 1+1):  # coordinator只有1个，所以r=1
            proc_type, proc_index = get_process_info(r, PROCESS_TYPES)
            rpc_info = rpc.get_worker_info(f"{proc_type}{proc_index}")
            rpc.rpc_sync(to=rpc_info, func=call_remote_method, args=(CacheCoordinator.set_workers_rref,self.coordinator_ref[0], self.worker_ref))
        time.sleep(1)
        print("[LLMScheduler] finish init all class")

    def add_prompt_list(self, prompt_list):
        self.prompt_list = prompt_list

    def process_prompt(self):
        futures = []
        for prompt in self.prompt_list:
            future = self._send_prompt(prompt)
            futures.append(future)
        
    def _send_prompt(self, prompt:InputPrompt):
        prompt_order = PromptOrder(prompt)
        # 历史优先，调度用户历史kvcache
        if prompt_order == "User History First":
            print(f"[LLMScheduler] Schedule a user history request")
            request_id = prompt.user_id
            send_cpu = self.strategy(prompt.user_id)
            task_info = (request_id, send_cpu)
            send_worker_ref = self.worker_ref[send_cpu]
            owner_worker_ref = send_worker_ref.owner()  
            rpc.rpc_sync(to=owner_worker_ref, func=call_remote_method, args=(Worker.receive_task_info, send_worker_ref, [task_info]))
        # 商品优先，调度*一组*商品kvcache
        elif prompt_order == "Item First":
            print(f"[LLMScheduler] Schedule a group of item request ({len(prompt.items)})")
            task_info_list = []
            for i in prompt.items:
                request_id = i.item_id
                send_cpu = self.strategy(i.item_id)
                task_info = (request_id, send_cpu) 
                task_info_list.append(task_info)
            # TODO 这里显然有问题！按照设计应该是发给不同的worker
            # 目前纯粹是取最后一个用到的worker，以后要改
            # 此外还需要解决cache是否miss的问题
            send_worker_ref = self.worker_ref[send_cpu]
            owner_worker_ref = send_worker_ref.owner() 
            rpc.rpc_sync(to=owner_worker_ref, func=call_remote_method, args=(Worker.receive_task_info, send_worker_ref, task_info_list))

    def strategy(self, user_id: int) -> int:
        return user_id % self.num_workers
    
    def set_prompt_generator(self, prompt_generator:LLMInput):
        self.prompt_generator = prompt_generator

    def start(self, iter_round:int, batchsize:int):
        # TODO 搞清楚到底是要做什么
        if not self.prompt_generator:
            print("[LLMScheduler] Error: prompt_generator IS None!")
            return
        for _ in range(iter_round):
            input_prompt_list = self.prompt_generator.Generate(batchsize)
            self.add_prompt_list(input_prompt_list)

def PromptOrder(prompt: InputPrompt) -> str:
    user_tokens = prompt.user_history_tokens
    item_tokens = sum([item.token_count for item in prompt.items])

    if user_tokens > item_tokens:
        return "User History First"
    else:
        return "Item First"
