from argparse import Namespace

from datasets import dataset_factory
from dataloader import LLMDataloader
import numpy as np
from typing import List, Dict, Any
import random

from dataloader.llm import LLMTestDataset

class LLMInput():
    def __init__(self,k:int,poisson_lambda:500,args:Namespace) -> None:
        self.k = -1
        self.args = args
        self.reset_k(k)
        self.poisson_lambda = poisson_lambda
        self.random_name = ""

    def Generate(self,batch_size: int) -> List[Dict]:
        prompts = []
        poisson_numbers = np.random.poisson(lam=self.poisson_lambda, size=batch_size)
        for ind,i in enumerate(self._get_random_index(batch_size)):
            data_point = self.dataset[i]
            user_id = data_point['user_id']
            user_history_tokens = data_point["history_length"] # 用户历史的token数量
            items = [{"item_id":data_point["candidates_id"][jnd],"token_count": len(j)} for jnd,j in enumerate(data_point["goods_index"])]
            timestamp = poisson_numbers[ind]  # 模拟timestamp
            prompts.append(InputPrompt(user_id,user_history_tokens,items,timestamp))
        return prompts
    
    def reset_k(self,k:int) -> None:
        self.k = k
        self.args.llm_negative_sample_size = k-1
        self.dataset = self._init_dataset()
    
    def set_random(self,random_name:str) -> None:
        random_name_list = ['sample','weighted']
        if random_name not in random_name_list:
            print("Warning: Invalid Random Name")
        self.random_name = random_name
    
    def _init_dataset(self) -> LLMTestDataset:
        dataset = dataset_factory(self.args)
        dataloader = LLMDataloader(self.args, dataset)
        llmDataset = dataloader._get_eval_dataset()
        return llmDataset
    
    def _get_random_index(self,batch_size: int) -> List[int]:
        indices = list(range(len(self.dataset)))
        if self.random_name == "sample":
            return random.sample(indices, batch_size)
        elif self.random_name == "weighted":
            weights = [data_point["history_length"] for data_point in self.dataset]
            sampled_index = random.choices(indices, weights=weights, k=batch_size)
            return sampled_index
        else:
            return list(range(batch_size))
        
class InputPrompt():
    def __init__(self,user_id:int,user_history_tokens:int,items:List[dict[str,Any]],timestamp:int) -> None:
        self.user_id = user_id
        self.user_history_tokens = user_history_tokens
        self.items = items
        self.timestamp = timestamp