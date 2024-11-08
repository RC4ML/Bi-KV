import random
from typing import List, Dict, Tuple

class LLMInput:
    @staticmethod
    def Generate(batch_size: int) -> List[Dict]:
        # 生成一批空prompt，模拟用户历史和商品的token数量
        prompts = []
        for i in range(batch_size):
            user_history_tokens = random.randint(5, 50)  # 用户历史的token数量
            num_products = random.randint(1, 10)  # 商品数量
            products = [{"token_count": random.randint(5, 20)} for _ in range(num_products)]
            timestamp = random.randint(1, 1000)  # 模拟timestamp
            prompts.append({
                "user_history_tokens": user_history_tokens,
                "products": products,
                "timestamp": timestamp
            })
        return prompts


class LLMModel:
    @staticmethod
    def PredictKVSize(prompt: Dict) -> int:
        # 假设每个token产生的KV Cache大小为1
        user_kv_size = prompt["user_history_tokens"]
        products_kv_size = sum([product["token_count"] for product in prompt["products"]])
        return user_kv_size + products_kv_size


class LLMScheduler:
    def __init__(self):
        self.prompts = []

    def schedule_prompts(self, batch_size: int):
        # 调用LLMInput.Generate生成prompt
        self.prompts = LLMInput.Generate(batch_size)

        # 仿真生成的每个prompt的KV大小
        for prompt in self.prompts:
            kv_size = LLMModel.PredictKVSize(prompt)
            prompt_order = self.PromptOrder(prompt)
            print(f"Prompt Timestamp: {prompt['timestamp']}, KV Size: {kv_size}, Prompt Order: {prompt_order}")

    def PromptOrder(self, prompt: Dict) -> str:
        # 判断用户历史在前还是商品在前
        # 这里简单假设用户历史token数大于总商品token数时，用户历史在前
        user_tokens = prompt["user_history_tokens"]
        product_tokens = sum([product["token_count"] for product in prompt["products"]])

        if user_tokens > product_tokens:
            return "User History First"
        else:
            return "Product First"


# 使用示例
scheduler = LLMScheduler()
scheduler.schedule_prompts(batch_size=5)
