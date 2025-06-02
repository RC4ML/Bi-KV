from argparse import Namespace
import logging

from datasets import dataset_factory
from dataloader import LLMDataloader
import numpy as np
from typing import List, Dict, Any
import random
import heapq
from collections import deque

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
        self.miss_user_history_tokens = 0
        self.miss_item_tokens = 0

class LLMInput():
    def __init__(self,k:int,poisson_lambda:500,args:Namespace,user_expand_ratio=1,use_dataloader=True) -> None:
        self.k = -1
        self.args = args
        self.poisson_lambda = poisson_lambda
        self.random_name = ""
        self.user_expand_ratio = user_expand_ratio
        if use_dataloader:
            self.reset_k(k)
        else:
            logging.warning("Using LLMInput without dataloader, this may cause issues with dataset initialization.")
            self.k = k

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
    
    def generate_time_series_without_dataloader(self,batch_size:int,timestep:int,time_step_map,user_item_map,user_token_map,item_token_map) -> List[InputPrompt]:
        prompts = []
        user_list = time_step_map[str(timestep)]
        sampling_weight = [i[1] for i in user_list]
        user_list = random.choices(user_list, k=batch_size, weights=sampling_weight)
        for i in range(batch_size):
            data_ind = user_list[i][0]
            item_list = user_item_map[data_ind]
            user_history_tokens = user_token_map[str(data_ind+2000000)]
            items = [PromptItem(item_id, item_token_map[str(item_id)]) for item_id in item_list]
            timestamp = timestep
            prompt = InputPrompt(data_ind,user_history_tokens,items,timestamp,0)
            prompts.append(prompt)
        random.shuffle(prompts)
        return prompts

    def generate_time_series_repeat_sampling(self,batch_size:int,timestep:int,time_step_map,user_item_map,user_token_map,item_token_map) -> List[InputPrompt]:
        """
        返回长度恰好 == final_batch_size 的 prompt 列表。
        1) 先抽 sample_user_cnt 个二元组；复制 => candidate_prompts
        2) 如 candidate_prompts > final_batch_size，则随机裁剪
        若不足，则继续按同权补充随机用户，或直接报错（取决于你的业务需求）
        3) 再用『最大堆+冷却』算法排布，使同用户间隔≥K
        """
        
        sample_user_cnt = batch_size
        final_batch_size = batch_size
        K = 8
        # ---------- 采样 ----------
        raw_pairs   = time_step_map[str(timestep)]
        print(len(raw_pairs))
        weights     = [freq for _,freq in raw_pairs]
        sampled     = random.choices(raw_pairs, k=sample_user_cnt, weights=weights)

        # ---------- 汇总频次 ----------
        freq = {}
        for uid,f in sampled:
            freq[uid] = freq.get(uid,0)+f

        # ---------- 生成“候选 prompt” ----------
        candidate_prompts = []
        for uid, cnt in freq.items():
            items_ids   = user_item_map[uid]
            items       = [PromptItem(it, item_token_map[str(it)]) for it in items_ids]
            user_tokens = user_token_map[str(uid+2000000)]
            for _ in range(cnt):
                candidate_prompts.append(InputPrompt(uid,user_tokens,items,timestep,0))

        # ---------- 根据需要裁剪 / 补充 ----------
        if len(candidate_prompts) > final_batch_size:
            candidate_prompts = random.sample(candidate_prompts, final_batch_size)
        elif len(candidate_prompts) < final_batch_size:
            # 这里给出一种“随机补足”方案；也可以选择 raise ValueError
            deficit = final_batch_size - len(candidate_prompts)
            extras  = random.choices(candidate_prompts, k=deficit)   # 简单复制已有
            candidate_prompts.extend(extras)

        # ---------- 最大堆 + 冷却队列排布 ----------
        # 统计缩减后每个用户剩余数量
        remain = {}
        for p in candidate_prompts:
            remain[p.user_id] = remain.get(p.user_id,0)+1
        heap = [(-cnt, uid) for uid,cnt in remain.items()]
        heapq.heapify(heap)
        cooldown_q = deque()
        seq, pos = [], 0
        # 再把 user_id → 对应所有 prompt 队列
        bucket = {}
        for p in candidate_prompts:
            bucket.setdefault(p.user_id, []).append(p)

        while heap or cooldown_q:
            while cooldown_q and cooldown_q[0][0] <= pos:
                _, neg, uid = cooldown_q.popleft()
                if neg < 0:
                    heapq.heappush(heap, (neg, uid))

            if heap:
                neg, uid = heapq.heappop(heap)
                prompt   = bucket[uid].pop()          # 取一条
                seq.append(prompt)

                neg += 1
                cooldown_q.append((pos+K+1, neg, uid))
            else:
                # 如果真的排不出，只能插空位；这里直接插一个随机 prompt
                seq.append(random.choice(candidate_prompts))
            pos += 1

        # 去掉多余占位，只保留最前面的 final_batch_size
        return seq[:final_batch_size]

    def generate_time_series_repeat_sampling_new(
            self,
            batch_size      : int,
            timestep        : int,
            time_step_map,
            user_item_map,
            user_token_map,
            item_token_map
    ) -> List[InputPrompt]:

        K = 8
        final_batch_size = batch_size
        sample_user_cnt  = batch_size

        # ---------- 采样 ----------
        raw_pairs = time_step_map[str(timestep)]
        weights   = [freq for _, freq in raw_pairs]
        sampled   = random.choices(raw_pairs, k=sample_user_cnt, weights=weights)

        # ---------- 汇总频次 ----------
        freq = {}
        for uid, f in sampled:
            freq[uid] = freq.get(uid, 0) + f

        # ---------- 生成候选 prompt ----------
        candidate_prompts = []
        for uid, cnt in freq.items():
            items_ids = user_item_map[uid]
            items     = [PromptItem(it, item_token_map[str(it)]) for it in items_ids]
            tokens    = user_token_map[str(uid + 2_000_000)]
            candidate_prompts.extend([
                InputPrompt(uid, tokens, items, timestep, 0) for _ in range(cnt)
            ])

        # 若多于 batch_size 先裁剪，少于就直接用（允许不足）
        if len(candidate_prompts) > final_batch_size:
            candidate_prompts = random.sample(candidate_prompts, final_batch_size)

        # ---------- 构造堆 ----------
        remain  = {}
        bucket  = {}
        for p in candidate_prompts:
            remain[p.user_id] = remain.get(p.user_id, 0) + 1
            bucket.setdefault(p.user_id, []).append(p)

        heap = [(-cnt, uid) for uid, cnt in remain.items()]
        heapq.heapify(heap)

        # ---------- 调度 ----------
        recent = deque(maxlen=K)       # 最近 K 个 user
        seq    = []

        while heap and len(seq) < final_batch_size:
            stash = []
            picked = None

            # 1) 找到一个不在 recent 的用户
            while heap:
                neg_cnt, uid = heapq.heappop(heap)
                if uid in recent:
                    stash.append((neg_cnt, uid))      # 暂存
                else:
                    picked = (neg_cnt, uid)
                    break

            # 2) 把暂存的全丢回堆
            for item in stash:
                heapq.heappush(heap, item)

            # 3) 如果没找到可用用户 → 约束无法满足，提前结束
            if picked is None:
                break

            # 4) 输出一条 prompt
            neg_cnt, uid = picked
            seq.append(bucket[uid].pop())     # 取一条
            recent.append(uid)

            neg_cnt += 1  # 少 1（注意负数）
            if neg_cnt < 0:
                heapq.heappush(heap, (neg_cnt, uid))

        return seq

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
        
    def access_index(self, index: int) -> InputPrompt:
        """Access a specific index in the dataset."""
        if 0 <= index < len(self.dataset):
            data_point = self.dataset[index]
            user_id = data_point['user_id']
            user_history_tokens = data_point["history_length"]*self.user_expand_ratio # 用户历史的token数量, NOTE: expand raw prompt length by 4x
            items = [PromptItem(data_point["candidates_id"][jnd],(len(j))) for jnd,j in enumerate(data_point["goods_index"])]
            return InputPrompt(user_id,user_history_tokens,items,0,0)
        else:
            raise IndexError("Index out of bounds for the dataset.")