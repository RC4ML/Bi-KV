import random
import numpy as np
from typing import List, Dict, Tuple

from dataloader.llm import LLMTestDataset

class LLMInput():
    def __init__(self,k:int,poisson_lambda:500,dataset:LLMTestDataset) -> None:
        self.k = k
        self.dataset = dataset
        self.poisson_lambda = poisson_lambda

    def Generate(self,batch_size: int) -> List[Dict]:
        # 生成一批空prompt，模拟用户历史和商品的token数量
        prompts = []
        poisson_numbers = np.random.poisson(lam=self.poisson_lambda, size=batch_size)
        for i in range(batch_size):
            user_history_tokens = self.dataset[i]["history_length"] # 用户历史的token数量
            items = [{"token_count": len(j)} for j in self.dataset[i]["goods_index"]]
            timestamp = poisson_numbers[i]  # 模拟timestamp
            prompts.append({
                "user_history_tokens": user_history_tokens,
                "items": items,
                "timestamp": timestamp
            })
        return prompts