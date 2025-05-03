# import uuid
import json
import random
from typing import Dict, List

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
        self.cache_coordinator_address = f"{rank_to_ip[1]}:{master_port+1}"
        self.coordinator_ref = []
        self.worker_ref = []
        self.prompt_list:List[InputPrompt] = []
        self.prompt_generator:LLMInput = None
        self._id_counter = 0
        self._task_counter = 0
        self.batchsize = 32
        self.cold_start_flag = True
        
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
            self._id_counter += 1
            prompt.task_id = self._id_counter
            working_prompt_list.append(prompt)
            cnt += 1
            if cnt % batch_size == 0:
                ans_dict = self._check_batch(working_prompt_list)
                self._send_prompt_batch(working_prompt_list,ans_dict)
                working_prompt_list = []
        # 处理尾巴部分（未整除的 prompts）
        remaining = total_prompts % batch_size  # 剩余未整除的数量
        if remaining > 0:  # 检查是否有剩余
            self._send_prompt_batch(self.prompt_list[-remaining:])
                        
    def _send_prompt_batch(self, prompt_list:List[InputPrompt],ans_dict):
        future_list = []
        task_info_list_dict = {}
        # 先check一遍cache
        for ind,prompt in enumerate(prompt_list): 
            prompt_order = PromptOrder(prompt,ans_dict)
            # 历史优先，调度用户历史kvcache
            if prompt_order == "User History First" or self.cold_start_flag:
            # if ans_dict[str(prompt.task_id)] == 1:
            # if True:
                infer_worker = self.strategy(prompt.task_id)
                token_num = prompt.user_history_tokens
                task_info = TaskInfo_pb2.TaskInfo(
                    request_id = prompt.task_id,
                    id = prompt.user_id+2000000,
                    infer_worker = infer_worker,
                    token_num = token_num,
                    index = 0,
                    task_type = SIGNAL_CHECK,
                    type = 'user cache',
                    task_num = 1,
                    weight=prompt.weight,
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
                    request_id = prompt.task_id,
                    id = -1,
                    infer_worker = infer_worker,
                    token_num = recomputing_tokens,
                    index = -1,
                    task_type = SIGNAL_SKIP,
                    type = 'compute',
                    task_num = 1,
                    weight=0,
                 )                                       
                task_info_list_dict[infer_worker].append(task_info)
                # TODO 冷启动flag位置

            # 商品优先，调度*一组*商品kvcache
            elif prompt_order == "Item First":
                # 商品优先级固定为0
                priority = 0
            # elif ans_dict[str(prompt.task_id)] == 0:
                infer_worker = self.strategy(prompt.task_id)
                # print(f"[LLMScheduler] Schedule a group of item request ({len(prompt.items)} to worker {infer_worker}, request id {self._id_counter})")
                for ind,i in enumerate(prompt.items):
                    token_num = i.token_count
                    task_info = TaskInfo_pb2.TaskInfo(
                        request_id = prompt.task_id,
                        id = i.item_id,
                        infer_worker = infer_worker,
                        token_num = token_num,
                        index = ind,
                        task_type = SIGNAL_CHECK,
                        type = 'item cache',
                        task_num = len(prompt.items),
                        weight=priority,
                    )
                    if task_info_list_dict.get(infer_worker):
                        task_info_list_dict[infer_worker].append(task_info)
                    else:
                        task_info_list_dict[infer_worker]=[task_info]
                ## append recomputing tokens
                task_info = TaskInfo_pb2.TaskInfo(
                    request_id = prompt.task_id,
                    id = -1,
                    infer_worker = infer_worker,
                    token_num = prompt.user_history_tokens,
                    index = -1,
                    task_type = SIGNAL_SKIP,
                    type = 'compute',
                    task_num = len(prompt.items),
                    weight=priority,
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
        
        # 控制写
        future_list = []
        for infer_worker in task_info_list_dict:
            infer_worker_port = self.master_port + 2*infer_worker + WORKER_offset
            # print(f"[LLMScheduler] Send task({len(task_info_list_dict[infer_worker])}) to worker {infer_worker} at port {infer_worker_port}")
            task_info_list = TaskInfo_pb2.TaskInfoList(tasks = task_info_list_dict[infer_worker]) 
            channel = grpc.insecure_channel(f"{self.rank_to_ip[2*infer_worker + WORKER_offset]}:{infer_worker_port}")
            stub = TaskInfo_pb2_grpc.InferWorkerServiceStub(channel)
            future = stub.StartWriteCacheData.future(task_info_list)
            future_list.append((future,channel))  
            
        for future,channel in future_list:
            future.result()
            channel.close()
        # 一批发送完后修改冷启动
        self.cold_start_flag = False

    def strategy(self, req_id: int) -> int:
        return req_id % self.num_workers
    
    def set_prompt_generator(self, prompt_generator:LLMInput):
        self.prompt_generator = prompt_generator

    def start(self, iter_round:int, batchsize:int, timestep_map = None):
        if not self.prompt_generator:
            print("[LLMScheduler] Error: prompt_generator is NONE!")
            return
        for timestep in range(iter_round):
            if timestep_map != None:
                input_prompt_list = self.prompt_generator.generate_time_series(batchsize,timestep,timestep_map)
            else:
                input_prompt_list = self.prompt_generator.generate(batchsize)
            self.add_prompt_list(input_prompt_list)
        # self.process_prompt()
        self.process_prompt_batch()
        # 在这之后调CacheCoordinator.send_terminate_signal，会炸，不知道为什么
        
    def calculate_data_len(self,token_num:int):
        return token_num*model_params["head_size"]*model_params["num_q_heads"]*model_params["num_layers"]*model_params["num_kv_heads"]
    
    def _check_batch(self,prompt_list:List[InputPrompt]):
        '''调度之前先查一遍cache'''
        send_list = []
        for prompt in prompt_list:
            # 用户优先
            token_num = prompt.user_history_tokens
            # 需要一个标识
            task_info = TaskInfo_pb2.TaskInfo(
                    request_id = prompt.task_id,
                    id = prompt.user_id+2000000,
                    token_num = token_num,
                    index = 0,
                    task_type = SIGNAL_CHECK,
                    type = 'user cache',
                    task_num = 1
                )
            send_list.append(task_info)
            # item优先
            for ind,i in enumerate(prompt.items):
                token_num = i.token_count
                task_info = TaskInfo_pb2.TaskInfo(
                        request_id = prompt.task_id,
                        id = i.item_id,
                        token_num = token_num,
                        index = ind,
                        task_type = SIGNAL_CHECK,
                        type = 'item cache',
                        task_num = len(prompt.items)
                    )
                send_list.append(task_info)
        # 查cache
        task_info_list = TaskInfo_pb2.TaskInfoList(tasks = send_list) 
        channel = grpc.insecure_channel(self.cache_coordinator_address)
        stub = TaskInfo_pb2_grpc.CacheCoordinatorServiceStub(channel)
        future = stub.ReceiveTasksFromScheduler.future(task_info_list)
        cache_miss_dict_data = future.result()
        # ans dict结构 Dict[req id]0/1
        ans_dict = json.loads(cache_miss_dict_data.msg)
        # print(f"ans_dict:{ans_dict}")
        return ans_dict

    def schedule_strategy(self):
        pass

def PromptOrder(prompt: InputPrompt,ans_dict = None) -> str:
    user_tokens = prompt.user_history_tokens
    item_tokens = sum([item.token_count for item in prompt.items])
    # print(f"user {user_tokens}, item {item_tokens}")
    if user_tokens >= item_tokens:
        if ans_dict != None:
            # 如果用户cache命中则用户优先，否则商品优先
            if ans_dict[str(prompt.task_id)]['user miss']==0:
                return "User History First"
            else:
                # 需要用cache空间判断
                return "Item First"
        return "User History First"
    else:
        return "Item First"
    
def dict_to_taskinfo(task_dict):
    task = TaskInfo_pb2.TaskInfo()
    ParseDict(task_dict, task)
    return task