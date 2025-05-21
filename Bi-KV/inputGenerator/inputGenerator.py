from argparse import Namespace

from datasets import dataset_factory
from dataloader import LLMDataloader
import numpy as np
from typing import List, Dict, Any
import random

from dataloader.llm import LLMTestDataset

class PromptItem():
    def __init__(self,item_id:int,token_count:int) -> None:
        self.item_id = item_id
        self.token_count = token_count
    def __str__(self) -> str:
        return f"{{item_id:{self.item_id}, token_count:{self.token_count}}}"
    def __repr__(self) -> str:
        return self.__str__()

class InputPrompt():
    def __init__(self,user_id:int,user_history_tokens:int,items:List[PromptItem],timestamp:int,weight:int) -> None:
        self.user_id = user_id
        self.user_history_tokens = user_history_tokens
        self.items = items
        self.timestamp = timestamp
        self.task_id = 0
        self.item_tokens = sum([i.token_count for i in self.items])
        self.weight = weight
        self.order = None

class LLMInput():
    def __init__(self,k:int,poisson_lambda:500,args:Namespace,user_expand_ratio=1) -> None:
        self.k = -1
        self.args = args
        self.reset_k(k)
        self.poisson_lambda = poisson_lambda
        self.random_name = ""
        self.user_expand_ratio = user_expand_ratio

    def generate(self,batch_size: int) -> List[InputPrompt]:
        prompts = []
        poisson_numbers = np.random.poisson(lam=self.poisson_lambda, size=batch_size)
        for ind,i in enumerate(self._get_random_index(batch_size)):
            data_point = self.dataset[i]
            user_id = data_point['user_id']
            user_history_tokens = data_point["history_length"]*self.user_expand_ratio # 用户历史的token数量, NOTE: expand raw prompt length by 4x
            items = [PromptItem(data_point["candidates_id"][jnd],(len(j))) for jnd,j in enumerate(data_point["goods_index"])]
            timestamp = poisson_numbers[ind]  # 模拟timestamp
            prompts.append(InputPrompt(user_id,user_history_tokens,items,timestamp,0))
        return prompts
    
    def generate_time_series(self,batch_size:int,timestep:int,time_step_map) -> List[InputPrompt]:
        '''根据时序数据产生batch'''
        prompts = []
        user_list = time_step_map[str(timestep)]
        sampling_weight = [i[1] for i in user_list]
        user_list = random.choices(user_list, k=batch_size, weights=sampling_weight)
        for i in range(batch_size):
            data_ind = user_list[i][0]
            # 时间步内访问次数
            access_count = user_list[i][1]
            data_point = self.dataset[data_ind]
            user_id = data_point['user_id']
            user_history_tokens = data_point["history_length"]*self.user_expand_ratio # 用户历史的token数量, NOTE: expand raw prompt length by 4x
            items = [PromptItem(data_point["candidates_id"][jnd],(len(j))) for jnd,j in enumerate(data_point["goods_index"])]
            timestamp = timestep  # 模拟timestamp
            weight = access_count * user_history_tokens
            prompt = InputPrompt(user_id,user_history_tokens,items,timestamp,weight)
            # for _ in range(access_count):
            prompts.append(prompt)
            # 很奇怪，这样会卡住
            # prompts.extend([prompt]*access_count)
            random.shuffle(prompts)
        return prompts
    
    def reset_k(self,k:int) -> None:
        self.k = k
        self.args.llm_negative_sample_size = k-1
        self.dataset = self._init_dataset()
    
    def set_random(self,random_name:str) -> None:
        random_name_list = ['random','weighted']
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
        if self.random_name == "random":
            return random.sample(indices, batch_size)
        elif self.random_name == "weighted":
            weights = [data_point["history_length"] for data_point in self.dataset]
            sampled_index = random.choices(indices, weights=weights, k=batch_size)
            return sampled_index
        else:
            return list(range(batch_size))