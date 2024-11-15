from argparse import Namespace
from datasets import dataset_factory
from dataloader import LLMDataloader
import numpy as np
from typing import List, Dict

from dataloader.llm import LLMTestDataset

class LLMInput():
    def __init__(self,k:int,poisson_lambda:500,args:Namespace) -> None:
        self.k = -1
        self.args = args
        self.reset_k(k)
        self.poisson_lambda = poisson_lambda

    def Generate(self,batch_size: int) -> List[Dict]:
        prompts = []
        poisson_numbers = np.random.poisson(lam=self.poisson_lambda, size=batch_size)
        for i in range(batch_size):
            data_point = self.dataset[i]
            user_id = data_point['user_id']
            user_history_tokens = data_point["history_length"] # 用户历史的token数量
            items = [{"item_id":data_point["candidates_id"][jnd],"token_count": len(j)} for jnd,j in enumerate(data_point["goods_index"])]
            timestamp = poisson_numbers[i]  # 模拟timestamp
            prompts.append({
                "user_id": user_id,
                "user_history_tokens": user_history_tokens,
                "items": items,
                "timestamp": timestamp
            })
        return prompts
    
    def reset_k(self,k:int) -> None:
        self.k = k
        self.args.llm_negative_sample_size = k-1
        self.dataset = self._init_dataset()
    
    def _init_dataset(self) -> LLMTestDataset:
        dataset = dataset_factory(self.args)
        dataloader = LLMDataloader(self.args, dataset)
        llmDataset = dataloader._get_eval_dataset()
        return llmDataset