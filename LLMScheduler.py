import random
from typing import List, Dict, Tuple
from config import *
from datasets import dataset_factory
from dataloader import LLMDataloader
from inputGenerator.inputGenerator import LLMInput
from Storage import KVCache

# from huggingface_hub import login
# login()
args.model_code = 'llm'
args.llm_retrieved_path = "/share/gnn_data/testmodel/LlamaRec/experiments/lru/games/"
args.dataset_code = "games"
set_template(args)

class LLMScheduler:
    def __init__(self, model_params: Dict):
        self.prompts = []
        total_capacity = 20000000000  # 20GB
        user_cache_ratio = 0.5
        model_layers = model_params.get("num_layers")
        vector_dim = model_params.get("head_size") * model_params.get("num_kv_heads")
        self.cache = KVCache(total_capacity=total_capacity, user_cache_ratio=user_cache_ratio, model_layers=model_layers, vector_dim=vector_dim)
        self.llm_input = LLMInput(20,500,args)

        
    def schedule_prompts(self, batch_size: int):

        generate_res = self.llm_input.Generate(batch_size)

        self.prompts = generate_res
        user_access_time = 0
        item_access_time = 0
        user_cache_miss_times = 0
        item_cache_miss_times = 0
        computation_cost = 0
        for ind, prompt in enumerate(self.prompts):
            prompt_order = self.PromptOrder(prompt)
            if prompt_order == "User History First":
                user_access_time += 1
                user_data = self.cache.get(cache_type='user', key=ind)
                computation_cost += sum([item["token_count"] for item in prompt["items"]])
                if user_data is None:
                    user_cache_miss_times += 1
                    self.cache.put(cache_type='user', key=ind, sequence_length=prompt["user_history_tokens"])
                    computation_cost += prompt["user_history_tokens"]
                # print("Current user cache size:", self.cache.user_cache.current_size)
                # print("Current user cache keys:", list(self.cache.user_cache.cache.keys()))
                # print("-" * 50)
            else:
                assert len(prompt['items']) == 20
                item_access_time += 20
                computation_cost += prompt["user_history_tokens"]
                # print("item first")
                for i, item in enumerate(prompt["items"]):
                    item_data = self.cache.get(cache_type='item', key=i+10000*ind)
                    if item_data is None:
                        item_cache_miss_times += 1
                        self.cache.put(cache_type='item', key=i+10000*ind, sequence_length=item["token_count"])
                        computation_cost += item["token_count"]
            # user_kv_size_mb = user_kv_size / (1024 * 1024)
            # item_kv_sizes_mb = [size / (1024 * 1024) for size in item_kv_sizes]
            # print(f"Prompt Timestamp: {prompt['timestamp']}, User KV Size: {user_kv_size_mb:.2f} MB, "
            #       f"Item KV Sizes: {[f'{size:.2f} MB' for size in item_kv_sizes_mb]}, Prompt Order: {prompt_order}")
        print("User Cache Hit Rate:", 1-user_cache_miss_times/(user_access_time))
        print("Item Cache Hit Rate:", 1-item_cache_miss_times/(item_access_time))
        print("Computation Cost: {} tokens".format(computation_cost))
        
    def PromptOrder(self, prompt: Dict) -> str:
        user_tokens = prompt["user_history_tokens"]
        item_tokens = sum([item["token_count"] for item in prompt["items"]])

        if user_tokens > item_tokens:
            return "User History First"
        else:
            return "Item First"


# Qwen2 1.5B
model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

scheduler = LLMScheduler(model_params)

scheduler.schedule_prompts(batch_size=10)
scheduler.schedule_prompts(batch_size=20)
