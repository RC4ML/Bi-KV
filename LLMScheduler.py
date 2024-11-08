import random
from typing import List, Dict, Tuple

class LLMInput:
    @staticmethod
    def Generate(batch_size: int) -> List[Dict]:
        # 生成一批空prompt，模拟用户历史和商品的token数量
        prompts = []
        for i in range(batch_size):
            user_history_tokens = random.randint(5, 50)  # 用户历史的token数量
            num_items = random.randint(1, 10)  # 商品数量
            items = [{"token_count": random.randint(5, 20)} for _ in range(num_items)]
            timestamp = random.randint(1, 1000)  # 模拟timestamp
            prompts.append({
                "user_history_tokens": user_history_tokens,
                "items": items,
                "timestamp": timestamp
            })
        return prompts
    
class LLMModel:
    def __init__(self, model_params: Dict):
        """
        model_params: 包含LLM模型参数的字典
        """
        self.model_params = model_params

    def PredictKVSize(self, prompt: Dict) -> Tuple[int, List[int]]:
        # 基于模型参数计算每个token的KV Cache大小
        kv_size_per_token = self.calculate_kv_size_per_token()

        # 计算用户历史的KV大小
        user_kv_size = prompt["user_history_tokens"] * kv_size_per_token

        # 计算每个item的KV大小
        item_kv_sizes = [item["token_count"] * kv_size_per_token for item in prompt["items"]]

        return user_kv_size, item_kv_sizes

    def calculate_kv_size_per_token(self) -> int:
        # 使用模型参数计算每个token的KV Cache大小
        head_size = self.model_params.get("head_size")
        num_heads = self.model_params.get("num_kv_heads")
        num_layers = self.model_params.get("num_layers")

        kv_size_per_token = (head_size * num_heads) * num_layers * 2 # FP16
        return kv_size_per_token


class LLMScheduler:
    def __init__(self, model_params: Dict):
        self.prompts = []
        self.model = LLMModel(model_params)
        
    def schedule_prompts(self, batch_size: int):
        # 调用LLMInput.Generate生成prompt
        self.prompts = LLMInput.Generate(batch_size)

        # 仿真生成的每个prompt的KV大小
        for prompt in self.prompts:
            user_kv_size, item_kv_sizes = self.model.PredictKVSize(prompt)
            prompt_order = self.PromptOrder(prompt)
            # 在打印时将 KV 大小转换为 MB
            user_kv_size_mb = user_kv_size / (1024 * 1024)
            item_kv_sizes_mb = [size / (1024 * 1024) for size in item_kv_sizes]
            print(f"Prompt Timestamp: {prompt['timestamp']}, User KV Size: {user_kv_size_mb:.2f} MB, "
                  f"Item KV Sizes: {[f'{size:.2f} MB' for size in item_kv_sizes_mb]}, Prompt Order: {prompt_order}")
            
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

scheduler.schedule_prompts(batch_size=5)
