from typing import List
import torch.distributed.rpc as rpc

from inputGenerator.inputGenerator import InputPrompt

# Qwen2 1.5B
model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

class LLMScheduler:
    def __init__(self, worker_func, world_size:int):
        self.worker_func = worker_func
        self.world_size = world_size
        self.num_workers = world_size - 1
        self.strategy_mode = "Default"
    
    def start(self):
        options = rpc.TensorPipeRpcBackendOptions(init_method='tcp://localhost:29500', num_worker_threads=256)
        rpc.init_rpc(
            name="master",
            rank=0,
            world_size=self.world_size,
            rpc_backend_options=options
        )
        print("Master initialized")

    def schedule(self,prompt_list: List[InputPrompt]):
        self.prompts = prompt_list
        res = self._send_prompt(prompt_list)
        user_cache_miss_times = sum(i['user_cache_miss_times'] for i in res)
        item_cache_miss_times = sum(i['item_cache_miss_times'] for i in res)
        user_access_time = sum(i['user_access_time'] for i in res)
        item_access_time = sum(i['item_access_time'] for i in res)
        computation_cost = sum(i['computation_cost'] for i in res)
        print("User Cache Hit Rate:", 1-user_cache_miss_times/(user_access_time))
        print("Item Cache Hit Rate:", 1-item_cache_miss_times/(item_access_time))
        print("Computation Cost: {} tokens".format(computation_cost))
        
    def _send_prompt(self,prompt_list: List[InputPrompt])->List:
        # 发送请求到相应的 worker
        futures = []
        for ind,prompt in enumerate(prompt_list):
            target_worker = f"worker{self.strategy(prompt.user_id)}"
            user_rref = rpc.remote(target_worker, InputPrompt, args=(prompt.user_id, prompt.user_history_tokens,prompt.items,prompt.timestamp))
            future = rpc.rpc_async(target_worker, self.worker_func, args=(user_rref,ind))
            futures.append(future)
        # 收集结果
        results = [fut.wait() for fut in futures]
        return results
    
    def update_worker(self):
        # TODO dynamic world_size
        self.num_workers += 1

    def strategy(self,user_id:int)->int:
        # schedule stategy
        if self.strategy_mode == "WIP":
            pass
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