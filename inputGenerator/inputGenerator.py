import random
from typing import List, Dict, Tuple

from dataloader.llm import LLMTestDataset

class LLMInput():
    def __init__(self,k:int,dataset:LLMTestDataset) -> None:
        self.k = k
        self.dataset = dataset

    def Generate(self,batch_size: int) -> List[Dict]:
        # 生成一批空prompt，模拟用户历史和商品的token数量
        prompts = []
        for i in range(batch_size):
            user_history_tokens = random.randint(5, 50)  # 用户历史的token数量
            items = [{"token_count": len(j)} for j in self.dataset[i]["goods_index"]]
            timestamp = random.randint(1, 1000)  # 模拟timestamp
            prompts.append({
                "user_history_tokens": user_history_tokens,
                "items": items,
                "timestamp": timestamp
            })
        return prompts