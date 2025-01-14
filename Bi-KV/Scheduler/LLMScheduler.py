# import uuid
from typing import List

import torch.distributed.rpc as rpc
from inputGenerator.inputGenerator import LLMInput,InputPrompt
from Worker.Worker import Worker
from rpc_def import PROCESS_TYPES, WORKER_NUM, KVCACHE_NUM, get_process_info, KVCACHE_offset
from DistributedStorage.CacheCoordinator import CacheCoordinator, KVCache
from DistributedStorage.Signals import SIGNAL_SKIP, SIGNAL_CHECK, SIGNAL_CHECK
from Remote.remote_call import call_remote_method
import time

model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

class LLMScheduler:
    def __init__(self, world_size:int):
        self.world_size = world_size
        self.num_workers = WORKER_NUM
        self.coordinator_ref = []
        self.worker_ref = []
        self.kvcache_ref = []
        self.prompt_list = []
        self.prompt_generator:LLMInput = None
        self._id_counter = 0
        self.batchsize = 32
        
        # 获取coordinator的rpc info
        # 根据PROCESS_TYPES: scheduler=0, coordinator=1, workers=2...(2+WORKER_NUM-1), kvcache后面
        for r in range(1, 1+1):  # coordinator只有1个，所以r=1
            proc_type, proc_index = get_process_info(r, PROCESS_TYPES)
            rpc_info = rpc.get_worker_info(f"{proc_type}{proc_index}")
            # 创建CacheCoordinator实例，传入KVCACHE_NUM
            self.coordinator_ref.append(
                rpc.remote(to=rpc_info, func=CacheCoordinator, args=(r, KVCACHE_NUM))
            )
            print("[LLMScheduler]finish init coordinator")

        # 初始化worker
        start_worker_rank = 2
        end_worker_rank = 2 + WORKER_NUM
        for r in range(start_worker_rank, end_worker_rank):
            proc_type, proc_index = get_process_info(r, PROCESS_TYPES)
            rpc_info = rpc.get_worker_info(f"{proc_type}{proc_index}")
            self.worker_ref.append(
                rpc.remote(to=rpc_info, func=Worker, args=(r, self.coordinator_ref[0]))
            )
        # 将workers_rref同步到coordinator
        for r in range(1, 1+1):  # coordinator只有1个，所以r=1
            proc_type, proc_index = get_process_info(r, PROCESS_TYPES)
            rpc_info = rpc.get_worker_info(f"{proc_type}{proc_index}")
            self.set_worker_call = rpc.rpc_async(to=rpc_info, func=call_remote_method, 
                         args=(CacheCoordinator.set_workers_rref,self.coordinator_ref[0], self.worker_ref))
        time.sleep(1)
        print("[LLMScheduler] finish init all class")

    def add_prompt_list(self, prompt_list):
        self.prompt_list.extend(prompt_list)

    def process_prompt(self):
        future_list = []
        cnt = 0
        for prompt in self.prompt_list:
            # self._send_prompt(prompt)
            future_list.append(self._send_prompt(prompt))
            cnt += 1
            if cnt % (self.num_workers * self.batchsize) == 0:
                for future in future_list:
                    future.wait()
                future_list = []
        if len(future_list) > 0:    
            for future in future_list:
                future.wait()
                
    def process_prompt_batch(self):
        cnt = 0
        working_prompt_list = []
        batch_size = self.num_workers * self.batchsize  # 每批次的大小
        total_prompts = len(self.prompt_list)
        for prompt in self.prompt_list:
            working_prompt_list.append(prompt)
            cnt += 1
            if cnt % batch_size == 0:
                self._send_prompt_batch(working_prompt_list)
                working_prompt_list = []
        # 处理尾巴部分（未整除的 prompts）
        remaining = total_prompts % batch_size  # 剩余未整除的数量
        if remaining > 0:  # 检查是否有剩余
            self._send_prompt_batch(self.prompt_list[-remaining:])
                        
    def _send_prompt(self, prompt:InputPrompt):
        prompt_order = PromptOrder(prompt)
        # 历史优先，调度用户历史kvcache
        if prompt_order == "User History First":
            task_info_list_dict = {}
            infer_worker = self.strategy(prompt.user_id)
            token_num = prompt.user_history_tokens
            data_length = self.calculate_data_len(token_num) 
            self._id_counter += 1
            # print(f"[LLMScheduler] Schedule a user history request to worker {infer_worker}, request id {self._id_counter}")
            task_info = {"request_id":self._id_counter,
                         "id":prompt.user_id, 
                         "infer_worker":infer_worker, 
                         "token_num":token_num,
                         'data_length':data_length,
                         'task_type': SIGNAL_CHECK,
                         'index': 0,
                         'type': 'user cache'
                         }
            task_info_list_dict[infer_worker]=[task_info]
            ## append recomputing tokens
            recomputing_tokens = 0
            for ind,i in enumerate(prompt.items):
                recomputing_tokens = i.token_count
            task_info = {"request_id":self._id_counter,
                            "id":-1, 
                            "infer_worker":infer_worker, 
                            "token_num":recomputing_tokens,
                            "data_length":-1,
                            'index':-1,
                            'task_type': SIGNAL_SKIP,
                            'type':'compute'}                                            
            task_info_list_dict[infer_worker].append(task_info)
            infer_worker_ref = self.worker_ref[infer_worker]
            owner_worker_ref = infer_worker_ref.owner()
            return rpc.rpc_async(to=owner_worker_ref, func=call_remote_method, 
                         args=(Worker.receive_task_info, infer_worker_ref, task_info_list_dict[infer_worker]))
        # 商品优先，调度*一组*商品kvcache
        elif prompt_order == "Item First":
            task_info_list_dict = {}
            self._id_counter += 1
            infer_worker = self.strategy(prompt.user_id)
            # print(f"[LLMScheduler] Schedule a group of item request ({len(prompt.items)} to worker {infer_worker}, request id {self._id_counter})")
            for ind,i in enumerate(prompt.items):
                token_num = i.token_count
                data_length = self.calculate_data_len(token_num) 
                task_info = {"request_id":self._id_counter,
                             "id":i.item_id, 
                             "infer_worker":infer_worker, 
                             "token_num":token_num,
                             "data_length":data_length,
                             'index':ind,
                             'task_type': SIGNAL_CHECK,
                             'type':'item cache'}
                if task_info_list_dict.get(infer_worker):
                    task_info_list_dict[infer_worker].append(task_info)
                else:
                    task_info_list_dict[infer_worker]=[task_info]
            ## append recomputing tokens
            task_info = {"request_id":self._id_counter,
                "id":-1, 
                "infer_worker":infer_worker, 
                "token_num":prompt.user_history_tokens,
                "data_length":-1,
                'index':-1,
                'task_type': SIGNAL_SKIP,
                'type':'compute'}
            task_info_list_dict[infer_worker].append(task_info)
            # TODO 还需要解决cache是否miss的问题
            if infer_worker in task_info_list_dict:
                infer_worker_ref = self.worker_ref[infer_worker]
                owner_worker_ref = infer_worker_ref.owner() 
                return rpc.rpc_async(to=owner_worker_ref, func=call_remote_method, 
                            args=(Worker.receive_task_info, infer_worker_ref, task_info_list_dict[infer_worker]))


    def _send_prompt_batch(self, prompt_list:List[InputPrompt]):
        future_list = []
        task_info_list_dict = {}

        for ind,prompt in enumerate(prompt_list): 
            prompt_order = PromptOrder(prompt)
            # 历史优先，调度用户历史kvcache
            if prompt_order == "User History First":
                infer_worker = self.strategy(self._id_counter)
                token_num = prompt.user_history_tokens
                data_length = self.calculate_data_len(token_num) 
                self._id_counter += 1
                # print(f"[LLMScheduler] Schedule a user history request to worker {infer_worker}, request id {self._id_counter}")
                task_info = {"request_id":self._id_counter,
                             # 可以用type来判断，不用+2000000
                            "id":prompt.user_id+2000000, # 避免user和item碰撞，user_id+2000000 
                            "infer_worker":infer_worker, 
                            "token_num":token_num,
                            'data_length':data_length,
                            'index': 0,
                            'task_type': SIGNAL_CHECK,
                            'type': 'user cache',
                            'task_num': 2
                            }
                if task_info_list_dict.get(infer_worker):
                    task_info_list_dict[infer_worker].append(task_info)
                else:
                    task_info_list_dict[infer_worker]=[task_info]
                ## append recomputing tokens
                recomputing_tokens = 0
                for ind,i in enumerate(prompt.items):
                    recomputing_tokens = i.token_count
                task_info = {"request_id":self._id_counter,
                                "id":-1, 
                                "infer_worker":infer_worker, 
                                "token_num":recomputing_tokens,
                                "data_length":-1,
                                'index':-1,
                                'task_type': SIGNAL_SKIP,
                                'type':'compute',
                                'task_num': 2}                                            
                task_info_list_dict[infer_worker].append(task_info)

            # 商品优先，调度*一组*商品kvcache
            elif prompt_order == "Item First":
                self._id_counter += 1
                infer_worker = self.strategy(self._id_counter)
                # print(f"[LLMScheduler] Schedule a group of item request ({len(prompt.items)} to worker {infer_worker}, request id {self._id_counter})")
                for ind,i in enumerate(prompt.items):
                    token_num = i.token_count
                    data_length = self.calculate_data_len(token_num) 
                    task_info = {"request_id":self._id_counter,
                                "id":i.item_id, 
                                "infer_worker":infer_worker, 
                                "token_num":token_num,
                                "data_length":data_length,
                                'index':ind,
                                'task_type': SIGNAL_CHECK,
                                'type':'item cache',
                                'task_num': (len(prompt.items)+1)}
                    if task_info_list_dict.get(infer_worker):
                        task_info_list_dict[infer_worker].append(task_info)
                    else:
                        task_info_list_dict[infer_worker]=[task_info]
                ## append recomputing tokens
                task_info = {"request_id":self._id_counter,
                    "id":-1, 
                    "infer_worker":infer_worker, 
                    "token_num":prompt.user_history_tokens,
                    "data_length":-1,
                    'index':-1,
                    'task_type': SIGNAL_SKIP,
                    'type':'compute',
                    'task_num': (len(prompt.items)+1)}
                task_info_list_dict[infer_worker].append(task_info)

        for infer_worker in task_info_list_dict:
            infer_worker_ref = self.worker_ref[infer_worker]
            owner_worker_ref = infer_worker_ref.owner() 
            future = rpc.rpc_async(to=owner_worker_ref, func=call_remote_method, 
                    args=(Worker.receive_task_info_batch, infer_worker_ref, task_info_list_dict[infer_worker]))
            future_list.append(future)  
            
        for future in future_list:
            future.wait()

    def strategy(self, req_id: int) -> int:
        return req_id % self.num_workers
    
    def set_prompt_generator(self, prompt_generator:LLMInput):
        self.prompt_generator = prompt_generator

    def start(self, iter_round:int, batchsize:int):
        if not self.prompt_generator:
            print("[LLMScheduler] Error: prompt_generator is NONE!")
            return
        for _ in range(iter_round):
            input_prompt_list = self.prompt_generator.Generate(batchsize)
            self.add_prompt_list(input_prompt_list)
        # self.process_prompt()
        self.process_prompt_batch()
        self.set_worker_call.wait()
        # 在这之后调CacheCoordinator.send_terminate_signal，会炸，不知道为什么
        
    def calculate_data_len(self,token_num:int):
        return token_num*model_params["head_size"]*model_params["num_q_heads"]*model_params["num_layers"]*model_params["num_kv_heads"]
    
    def test_write_cache(self):
        print(f"[LLMScheduler] Write test start")
        cache_worker = 1
        infer_worker = 2
        simulate_task_info = {
            "request_id":42,
            "id":42, 
            "infer_worker":infer_worker,
            'cache_worker':cache_worker, 
            "token_num":42,
            "data_length": 1234,
            'index':0
        }
        
        print(f"[LLMScheduler] Start recv")
        recv_data_call = rpc.rpc_async(to=self.coordinator_ref[0].owner(), func=call_remote_method, 
                         args=(CacheCoordinator.test_write,self.coordinator_ref[0],simulate_task_info))
        print(f"[LLMScheduler] Start send")
        cache_worker_ref = self.worker_ref[cache_worker]
        owner_worker_ref = cache_worker_ref.owner() 
        send_data_call = rpc.rpc_async(to=owner_worker_ref, func=call_remote_method, 
                            args=(Worker.send_kvcache_data, cache_worker_ref, simulate_task_info))
        send_data_call.wait()
        recv_data_call.wait()
        print(f"[LLMScheduler] Write test success!")
        


def PromptOrder(prompt: InputPrompt) -> str:
    user_tokens = prompt.user_history_tokens
    item_tokens = sum([item.token_count for item in prompt.items])
    # print(f"user {user_tokens}, item {item_tokens}")
    if user_tokens > item_tokens:
        return "User History First"
    else:
        return "Item First"
