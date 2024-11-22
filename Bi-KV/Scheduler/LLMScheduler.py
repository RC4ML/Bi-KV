import random
from typing import List, Dict, Tuple
import torch.distributed.rpc as rpc

from config import *
from datasets import dataset_factory
from dataloader import LLMDataloader
from inputGenerator.inputGenerator import InputPrompt, LLMInput
from Storage import KVCache

# from huggingface_hub import login
# login()
args.model_code = 'llm'
args.llm_retrieved_path = "/share/gnn_data/testmodel/LlamaRec/experiments/lru/games/"
args.dataset_code = "games"
set_template(args)

class LLMScheduler:
    def __init__(self, model_params: Dict, item_num,num_workers:3):
        self.prompts = []
        total_capacity = 20000000000  # 20GB
        user_cache_ratio = 0.5
        model_layers = model_params.get("num_layers")
        vector_dim = model_params.get("head_size") * model_params.get("num_kv_heads")
        self.cache = KVCache(total_capacity=total_capacity, user_cache_ratio=user_cache_ratio, model_layers=model_layers, vector_dim=vector_dim)
        self.llm_input = LLMInput(item_num,500,args)
        self.item_num = item_num
        self.num_workers = num_workers
        self.workers = [[] for _ in range(num_workers)]

        
    def schedule_prompts_example(self, prompt_list: List[InputPrompt]):
        self.prompts = prompt_list
        user_access_time = 0
        item_access_time = 0
        user_cache_miss_times = 0
        item_cache_miss_times = 0
        computation_cost = 0
        for ind, prompt in enumerate(self.prompts):
            prompt_order = self.PromptOrder(prompt)
            if prompt_order == "User History First":
                # print("User first")
                user_access_time += 1
                user_data = self.cache.get(cache_type='user', key=ind)
                computation_cost += sum([item.token_count for item in prompt.items])
                if user_data is None:
                    user_cache_miss_times += 1
                    self.cache.put(cache_type='user', key=ind, sequence_length=prompt.user_history_tokens)
                    computation_cost += prompt.user_history_tokens
                # print("Current user cache size:", self.cache.user_cache.current_size)
                # print("Current user cache keys:", list(self.cache.user_cache.cache.keys()))
                # print("-" * 50)
            else:
                assert len(prompt.items) == self.item_num
                item_access_time += self.item_num
                computation_cost += prompt.user_history_tokens
                # print("Item first")
                for i, item in enumerate(prompt.items):
                    item_data = self.cache.get(cache_type='item', key=i+10000*ind)
                    if item_data is None:
                        item_cache_miss_times += 1
                        self.cache.put(cache_type='item', key=i+10000*ind, sequence_length=item.token_count)
                        computation_cost += item.token_count
            # user_kv_size_mb = user_kv_size / (1024 * 1024)
            # item_kv_sizes_mb = [size / (1024 * 1024) for size in item_kv_sizes]
            # print(f"Prompt Timestamp: {prompt['timestamp']}, User KV Size: {user_kv_size_mb:.2f} MB, "
            #       f"Item KV Sizes: {[f'{size:.2f} MB' for size in item_kv_sizes_mb]}, Prompt Order: {prompt_order}")
        print("User Cache Hit Rate:", 1-user_cache_miss_times/(user_access_time))
        print("Item Cache Hit Rate:", 1-item_cache_miss_times/(item_access_time))
        print("Computation Cost: {} tokens".format(computation_cost))
        
    def PromptOrder(self, prompt: InputPrompt) -> str:
        user_tokens = prompt.user_history_tokens
        item_tokens = sum([item.token_count for item in prompt.items])

        if user_tokens > item_tokens:
            return "User History First"
        else:
            return "Item First"

    def _schedule(self,input:InputPrompt,worker_id:int) -> None:
        # TODO: 实现具体的worker
        self.workers[worker_id].append(input)

    def schedule_prompts(self,prompt_list: List[InputPrompt]) -> None:
        self.prompts = prompt_list
        for i in self.prompts:
            prompt_id = i.user_id
            worker_id = prompt_id%self.num_workers
            self._schedule(i,worker_id)

# Qwen2 1.5B
model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

# Scheduler 类
class Scheduler:
    def __init__(self, num_workers, worker_func,world_size):
        self.num_workers = num_workers
        self.worker_func = worker_func
        self.world_size = world_size
    
    def start(self):
        options = rpc.TensorPipeRpcBackendOptions(init_method='tcp://localhost:29500', num_worker_threads=256)
        rpc.init_rpc(
            name="master",
            rank=0,
            world_size=self.world_size,  # 假设有 2 个 worker 和 1 个 master
            rpc_backend_options=options
        )
        print("Master initialized")
        
        # 创建请求对象
        requests = [InputPrompt(user_id=i, user_history_tokens=i*10,items=[],timestamp=i*random.randint(1,100)) for i in range(1, 6)]
        
        # 发送请求到相应的 worker
        futures = []
        for req in requests:
            target_worker = f"worker{req.user_id % self.num_workers}"
            user_rref = rpc.remote(target_worker, InputPrompt, args=(req.user_id, req.user_history_tokens,req.items,req.timestamp))
            future = rpc.rpc_async(target_worker, self.worker_func, args=(user_rref,))
            futures.append(future)
        
        # 收集结果
        results = [fut.wait() for fut in futures]
        print(f"Results: {results}")
    
    def shutdown(self):
        rpc.shutdown()


if __name__ == "__main__":
    scheduler = LLMScheduler(model_params, 10)

    scheduler.schedule_prompts_example(batch_size=10)
    scheduler.schedule_prompts_example(batch_size=20)
