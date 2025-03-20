# import uuid
from typing import List

import grpc
from protos import TaskInfo_pb2,TaskInfo_pb2_grpc
from google.protobuf.json_format import ParseDict

from inputGenerator.inputGenerator import LLMInput,InputPrompt
from Worker.Worker import Worker
from rpc_def import *
from DistributedStorage.CacheCoordinator import CacheCoordinator, KVCache
from DistributedStorage.Signals import SIGNAL_SKIP, SIGNAL_CHECK, SIGNAL_CHECK
from Remote.remote_call import call_remote_method
import time

import torch.distributed.rpc as rpc

model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

class LLMScheduler:
    def __init__(self, world_size:int, master_port:int, rank_to_ip:dict):
        self.rank_to_ip = rank_to_ip
        self.world_size = world_size
        self.master_port = master_port
        self.num_workers = WORKER_NUM
        self.coordinator_ref = []
        self.worker_ref = []
        self.prompt_list = []
        self.prompt_generator:LLMInput = None
        self._id_counter = 0
        self.batchsize = 32
        
        time.sleep(1)
        print("[LLMScheduler] finish init all class")

    def add_prompt_list(self, prompt_list):
        self.prompt_list.extend(prompt_list)
                
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
                        
    def _send_prompt_batch(self, prompt_list:List[InputPrompt]):
        future_list = []
        task_info_list_dict = {}

        for ind,prompt in enumerate(prompt_list): 
            prompt_order = PromptOrder(prompt)
            # 历史优先，调度用户历史kvcache
            if prompt_order == "User History First":
                infer_worker = self.strategy(self._id_counter)
                token_num = prompt.user_history_tokens
                self._id_counter += 1
                task_info = TaskInfo_pb2.TaskInfo(
                    request_id = self._id_counter,
                    id = prompt.user_id+2000000,
                    infer_worker = infer_worker,
                    token_num = token_num,
                    index = 0,
                    task_type = SIGNAL_CHECK,
                    type = 'user cache',
                    task_num = 1
                )
                if task_info_list_dict.get(infer_worker):
                    task_info_list_dict[infer_worker].append(task_info)
                else:
                    task_info_list_dict[infer_worker]=[task_info]
                ## append recomputing tokens
                recomputing_tokens = 0
                for ind,i in enumerate(prompt.items):
                    recomputing_tokens = i.token_count 
                task_info = TaskInfo_pb2.TaskInfo(
                    request_id = self._id_counter,
                    id = -1,
                    infer_worker = infer_worker,
                    token_num = recomputing_tokens,
                    index = -1,
                    task_type = SIGNAL_SKIP,
                    type = 'compute',
                    task_num = 1
                 )                                       
                task_info_list_dict[infer_worker].append(task_info)

            # 商品优先，调度*一组*商品kvcache
            elif prompt_order == "Item First":
                self._id_counter += 1
                infer_worker = self.strategy(self._id_counter)
                # print(f"[LLMScheduler] Schedule a group of item request ({len(prompt.items)} to worker {infer_worker}, request id {self._id_counter})")
                for ind,i in enumerate(prompt.items):
                    token_num = i.token_count
                    task_info = TaskInfo_pb2.TaskInfo(
                        request_id = self._id_counter,
                        id = i.item_id,
                        infer_worker = infer_worker,
                        token_num = token_num,
                        index = ind,
                        task_type = SIGNAL_CHECK,
                        type = 'item cache',
                        task_num = len(prompt.items)
                    )
                    if task_info_list_dict.get(infer_worker):
                        task_info_list_dict[infer_worker].append(task_info)
                    else:
                        task_info_list_dict[infer_worker]=[task_info]
                ## append recomputing tokens
                task_info = TaskInfo_pb2.TaskInfo(
                    request_id = self._id_counter,
                    id = -1,
                    infer_worker = infer_worker,
                    token_num = prompt.user_history_tokens,
                    index = -1,
                    task_type = SIGNAL_SKIP,
                    type = 'compute',
                    task_num = len(prompt.items)
                )
                task_info_list_dict[infer_worker].append(task_info)

        for infer_worker in task_info_list_dict:
            infer_worker_port = self.master_port + 2*infer_worker + WORKER_offset
            # print(f"[LLMScheduler] Send task({len(task_info_list_dict[infer_worker])}) to worker {infer_worker} at port {infer_worker_port}")
            task_info_list = TaskInfo_pb2.TaskInfoList(tasks = task_info_list_dict[infer_worker]) 
            channel = grpc.insecure_channel(f"{self.rank_to_ip[2*infer_worker + WORKER_offset]}:{infer_worker_port}")
            stub = TaskInfo_pb2_grpc.InferWorkerServiceStub(channel)
            future = stub.ReceiveTasksFromScheduler.future(task_info_list)
            future_list.append((future,channel))  
            
        for future,channel in future_list:
            future.result()
            channel.close()

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
    
def dict_to_taskinfo(task_dict):
    task = TaskInfo_pb2.TaskInfo()
    ParseDict(task_dict, task)
    return task
