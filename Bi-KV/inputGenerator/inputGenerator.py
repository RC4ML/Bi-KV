from argparse import Namespace

from datasets import dataset_factory
from dataloader import LLMDataloader
import numpy as np
from typing import List, Dict, Any
import random
import logging

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
    def __init__(self,user_id:int,user_history_tokens:int,items:List[PromptItem],timestamp:int) -> None:
        self.user_id = user_id
        self.user_history_tokens = user_history_tokens
        self.items = items
        self.timestamp = timestamp
        self.task_id = 0

class LLMInput():
    def __init__(self,k:int,poisson_lambda:500,args:Namespace) -> None:
        self.k = -1
        self.args = args
        self.reset_k(k)
        self.poisson_lambda = poisson_lambda
        self.random_name = ""

    def Generate(self,batch_size: int) -> List[InputPrompt]:
        prompts = []
        poisson_numbers = np.random.poisson(lam=self.poisson_lambda, size=batch_size)
        for ind,i in enumerate(self._get_random_index(batch_size)):
            data_point = self.dataset[i]
            user_id = data_point['user_id']
            user_history_tokens = data_point["history_length"] * 20 # 用户历史的token数量, NOTE: expand raw prompt length by 4x
            items = [PromptItem(data_point["candidates_id"][jnd],(len(j)*5)) for jnd,j in enumerate(data_point["goods_index"])]
            timestamp = poisson_numbers[ind]  # 模拟timestamp
            # logging.info(f"[LLMInput] Generate prompt {ind}: user_id={user_id}, user_history_tokens={user_history_tokens}")
            prompts.append(InputPrompt(user_id,user_history_tokens,items,timestamp))
        return prompts

    def generate_time_series(self,batch_size:int,timestep:int,time_step_map) -> List[InputPrompt]:
        '''根据时序数据产生batch'''
        prompts = []
        user_list = time_step_map[str(timestep)]
        # user_list = random.sample(user_list[:1024],batch_size)
        ind_range = min(len(user_list), 10240)
        weights = [user_list[i][1] for i in range(ind_range)]
        indices = list(range(ind_range))
        sampled_index = random.choices(indices, weights=weights, k=batch_size)
            
        # batch_counter = 0
        # for i in user_list:
        #     user_id = i[0]
        #     access_times = i[1]
        #     data_point = self.dataset[user_id]
        #     user_id = data_point['user_id']
        #     user_history_tokens = data_point["history_length"]*4 # 用户历史的token数量, NOTE: expand raw prompt length by 4x
        #     items = [PromptItem(data_point["candidates_id"][jnd],(len(j))) for jnd,j in enumerate(data_point["goods_index"])]
        #     prompt = InputPrompt(user_id,user_history_tokens,items,timestep)
        #     if batch_counter+access_times<batch_size:
        #         prompts.extend([prompt]*access_times)
        #         batch_counter+=access_times
        #     else:
        #         prompts.extend([prompt]*(batch_size-batch_counter))
        #         batch_counter = batch_size-1
        #     if batch_counter == batch_size-1:
        #         break
        # print(prompts)
        for i in range(batch_size):
            data_ind = sampled_index[i]
            # data_ind = user_list[i][0]
            # access_count = user_list[i][1]
            data_point = self.dataset[data_ind]
            user_id = data_point['user_id']
            user_history_tokens = data_point["history_length"]*20 # 用户历史的token数量, NOTE: expand raw prompt length by 4x
            items = [PromptItem(data_point["candidates_id"][jnd],(len(j)*5)) for jnd,j in enumerate(data_point["goods_index"])]
            timestamp = timestep  # 模拟timestamp
            prompt = InputPrompt(user_id, user_history_tokens, items, timestamp)
            # for _ in range(access_count):
            prompts.append(prompt)
            # 很奇怪，这样会卡住
            # prompts.extend([prompt]*access_count)
        
        random.shuffle(prompts)
        return prompts

    def resample_prompts(self, prompts, chunk_size=10):
        """
        依据用户历史长度对 prompts 进行重采样并在最终顺序中交错分块，
        使得:
        1) 用户出现次数 ∝ 历史长度;
        2) 同一用户的条目尽量彼此靠近但不是一个大块;
        3) 不同用户的条目相互穿插。
        参数:
        prompts: 原始的 Prompt 列表，每个元素都有 user_id, user_history_tokens 等信息
        chunk_size: 进行二次分块的大小，可根据实际需求调参
        返回:
        new_prompts: 处理后的新 Prompt 列表
        """
        from collections import defaultdict
        import random
        
        # 1. 按 user_id 分组保存在 user2prompts
        user2prompts = defaultdict(list)
        for p in prompts:
            user2prompts[p.user_id].append(p)
        
        # 2. 计算每个用户的重复倍数（示例: 整除 100, 至少为 1）
        user2times = {}
        for user_id, p_list in user2prompts.items():
            # 这里用所有该用户条目的最大 history 作为参考
            max_history = max(pp.user_history_tokens for pp in p_list)
            times = max(1, max_history // 100)  
            user2times[user_id] = times
        
        # 3. 按顺序复制用户的 prompt 列表，再切分成小块
        user2chunks = {}
        for user_id, p_list in user2prompts.items():
            t = user2times[user_id]
            # 复制 times 倍
            repeated_prompts = p_list * t  
            
            # 将大的 repeated_prompts 切分为若干小块
            chunks = []
            for i in range(0, len(repeated_prompts), chunk_size):
                chunk = repeated_prompts[i : i + chunk_size]
                chunks.append(chunk)
            user2chunks[user_id] = chunks
        
        # 4. 按照 round-robin 的方式在“层次”上交错合并这些小块
        #    每一层取所有用户的第 i 个 chunk，先洗牌一下用户顺序再拼接
        max_num_chunks = max(len(chunks) for chunks in user2chunks.values())  # 所有用户子块数目的最大值
        final_sequence = []
        
        for layer_idx in range(max_num_chunks):
            # 当前层次哪些用户还有第 layer_idx 个子块
            candidate_users = [u for u, c in user2chunks.items() if len(c) > layer_idx]
            random.shuffle(candidate_users)  # 在这一层对用户做一个轻度洗牌
            
            # 将这一层所有用户的子块依次加到 final_sequence
            for u in candidate_users:
                final_sequence.extend(user2chunks[u][layer_idx])
                
        return final_sequence

    def Generate_test(self, batch_size: int, iter_round: int) -> List[InputPrompt]:
        prompts = []
        for ind,i in enumerate(self._get_random_index(batch_size)):
            data_point = self.dataset[i+int(iter_round/4*batch_size)]
            user_id = data_point['user_id']
            user_history_tokens = data_point["history_length"] * 20
            items = [PromptItem(data_point["candidates_id"][jnd],(len(j)*5)) for jnd,j in enumerate(data_point["goods_index"])]
            timestamp = 0
            prompts.append(InputPrompt(user_id,user_history_tokens,items,timestamp))
        
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
            if batch_size > len(self.dataset):
                logging.info("Warning: batch size is larger than dataset size, using all indices")
                assert (0)
            return list(range(batch_size))