from typing import List
import random
import torch.distributed.rpc as rpc

from inputGenerator.inputGenerator import InputPrompt
from Worker.Worker import Worker
from rpc_def import *
# Qwen2 1.5B
model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

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
        self.coordinator_ref =[]
        self.worker_ref =[]
        self.kvcache_ref =[]
        self.prompt_list = []
        print("[LLMScheduler]finish init LLMscheduler")

    def add_prompt_list(self,prompt_list):
        self.prompt_list=prompt_list
        # user_cache_miss_times = sum(i['user_cache_miss_times'] for i in res)
        # item_cache_miss_times = sum(i['item_cache_miss_times'] for i in res)
        # user_access_time = sum(i['user_access_time'] for i in res)
        # item_access_time = sum(i['item_access_time'] for i in res)
        # computation_cost = sum(i['computation_cost'] for i in res)
        # print("User Cache Hit Rate:", 1-user_cache_miss_times/(user_access_time))
        # print("Item Cache Hit Rate:", 1-item_cache_miss_times/(item_access_time))
        # print("Computation Cost: {} tokens".format(computation_cost))

    # def add_prompt(self,prompt:InputPrompt):
    #     self.prompt_list.append(prompt)
    #     print(f"[LLMScheduler] Add prompt {prompt.user_id} to list.")

    def process_prompt(self):
        futures = []
        for prompt in self.prompt_list:
            future = self._send_prompt(prompt)
            futures.append(future)
        # 收集结果
        # results = [fut.wait() for fut in futures]
        # return results
        
    def _send_prompt(self,prompt):
        # 发送请求到相应的 worker
        # worker_id = self.strategy(prompt.user_id)
        request_id, send_gpu, recv_gpu = prompt
        worker_id=recv_gpu  #这里都是接收方GPU和worker在同一张卡上
        send_worker_ref=self.worker_ref[worker_id]
        owner_worker_ref = send_worker_ref.owner()  # 获取 RRef 的拥有者
        future = rpc.rpc_sync(to=owner_worker_ref, func=_call_Worker_method, args=(Worker.receive_task_info,send_worker_ref, prompt),timeout=1)
        future.wait()
        return future
    
    # def update_worker(self):
    #     # TODO dynamic world_size
    #     self.num_workers += 1

    def strategy(self,user_id:int)->int:
        # schedule stategy
        if self.strategy_mode == "WIP":
            return random.randint(self.num_workers)
        else:
            return user_id%self.num_workers

    def shutdown(self):
        rpc.shutdown()
    
def PromptOrder(prompt: InputPrompt) -> str:
    user_tokens = prompt.user_history_tokens
    item_tokens = sum([item.token_count for item in prompt.items])

    if user_tokens > item_tokens:
        return "User History First"
    else:
        return "Item First"