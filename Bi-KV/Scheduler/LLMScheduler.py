# import uuid
import json
import logging
import random
from typing import Dict, List

import grpc
from protos import TaskInfo_pb2,TaskInfo_pb2_grpc
from google.protobuf.json_format import ParseDict

from inputGenerator.inputGenerator import LLMInput,InputPrompt
from Worker.Worker import Worker
from Utils.channelpool import ChannelPool
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
    def __init__(self, world_size:int, master_port:int, rank_to_ip:dict,max_batch_token:4000,batchsize:100):
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
        self.max_batch_token = max_batch_token
        self.batchsize = batchsize
        self.cold_start_flag = True
        self.channelpool = ChannelPool()
        
        time.sleep(1)
        print("[LLMScheduler] finish init all class")

    def add_prompt_list(self, prompt_list:List[InputPrompt]):
        for i in prompt_list:
            self._id_counter += 1
            i.task_id = self._id_counter
            self.prompt_list.append(i)
                
    def process_prompt_batch(self,hack_option = None):
        for i in range(0,len(self.prompt_list),self.batchsize):
            plan_tokens_num = 0
            working_prompt_list = []
            batch_list = self.prompt_list[i:i+self.batchsize]
            ans_dict = self._check_batch(batch_list)
            for ind, prompt in enumerate(batch_list):
                
                if hack_option == 'compete':
                    prompt.order = schedule_order_compete(prompt)
                elif hack_option != None:
                    prompt.order = hack_option
                else:
                    prompt.order = schedule_order_budget(prompt,self.max_batch_token)
                prompt.miss_user_history_tokens = ans_dict[str(prompt.task_id)]['user miss']
                prompt.miss_item_tokens = ans_dict[str(prompt.task_id)]['item miss']
                # UHF User前缀 item全重算
                if prompt.order == "User History First":
                    compute_token_num = prompt.item_tokens + prompt.miss_user_history_tokens
                elif prompt.order == "Item First":
                    compute_token_num = prompt.user_history_tokens + prompt.miss_item_tokens
                
                working_prompt_list.append(prompt)
                plan_tokens_num += compute_token_num
                if plan_tokens_num > self.max_batch_token:
                    # logging.info(f"[LLMScheduler] Plan tokens num: {plan_tokens_num}, max batch token: {self.max_batch_token} send batch {len(working_prompt_list)}")
                    self._send_prompt_batch(working_prompt_list,ans_dict)
                    working_prompt_list = []
                    plan_tokens_num = 0
            # 处理尾巴部分（未整除的 prompts）
            if len(working_prompt_list) > 0:  # 检查是否有剩余
                self._send_prompt_batch(working_prompt_list,ans_dict)
                        
    def _send_prompt_batch(self, prompt_list:List[InputPrompt],ans_dict = None,prepare_flag=False):
        future_list = []
        task_info_list_dict = {}
        # 先check一遍cache
        user_counter = 0
        item_counter = 0 
        total_counter = 0
        for ind,prompt in enumerate(prompt_list): 
            total_counter += 1
            # TODO 残留代码
            if prepare_flag:
                prompt.order = "Item First"
            elif self.cold_start_flag:
                prompt.order = "User History First"
            # else:
            #     prompt.order = schedule_order_budget(prompt, ans_dict)
            # prompt_order = schedule_order(prompt)
            # 历史优先，调度用户历史kvcache
            if prompt.order == "User History First":
            # if prompt_order == "User History First":
            # if ans_dict[str(prompt.task_id)] == 1:
            # if True:
                user_counter += 1
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
                    # NOTE weight为0为测试用
                    weight = 1,
                )
                if task_info_list_dict.get(infer_worker):
                    task_info_list_dict[infer_worker].append(task_info)
                else:
                    task_info_list_dict[infer_worker]=[task_info]
                ## append recomputing tokens
                # NOTE 重计算需要再次确认
                recomputing_tokens = 0
                for ind,i in enumerate(prompt.items):
                    recomputing_tokens += i.token_count 
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

            # 商品优先，调度*一组*商品kvcache
            elif prompt.order == "Item First":
                item_counter += 1
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
        # logging.info(f"[LLMScheduler] Send {total_counter} tasks, user {user_counter}, item {item_counter}")
        for infer_worker in task_info_list_dict:
            infer_worker_port = self.master_port + 2*infer_worker + WORKER_offset
            # print(f"[LLMScheduler] Send task({len(task_info_list_dict[infer_worker])}) to worker {infer_worker} at port {infer_worker_port}")
            task_info_list = TaskInfo_pb2.TaskInfoList(tasks = task_info_list_dict[infer_worker]) 
            channel = self.channelpool.get_channel(f"{self.rank_to_ip[2*infer_worker + WORKER_offset]}:{infer_worker_port}")
            stub = TaskInfo_pb2_grpc.InferWorkerServiceStub(channel)
            future = stub.ReceiveTasksFromScheduler.future(task_info_list)
            future_list.append((future,channel))  
            
        for future,channel in future_list:
            future.result()
        
        # 控制写
        write_future_list = []
        for infer_worker in task_info_list_dict:
            infer_worker_port = self.master_port + 2*infer_worker + WORKER_offset
            # print(f"[LLMScheduler] Send task({len(task_info_list_dict[infer_worker])}) to worker {infer_worker} at port {infer_worker_port}")
            task_info_list = TaskInfo_pb2.TaskInfoList(tasks = task_info_list_dict[infer_worker]) 
            channel = self.channelpool.get_channel(f"{self.rank_to_ip[2*infer_worker + WORKER_offset]}:{infer_worker_port}")
            stub = TaskInfo_pb2_grpc.InferWorkerServiceStub(channel)
            future = stub.StartWriteCacheData.future(task_info_list)
            write_future_list.append((future,channel))  
            
        for future,channel in write_future_list:
            future.result()
        # 一批发送完后修改冷启动
        if not prepare_flag:
            self.cold_start_flag = False

    def strategy(self, req_id: int) -> int:
        return req_id % self.num_workers
    
    def set_prompt_generator(self, prompt_generator:LLMInput):
        self.prompt_generator = prompt_generator

    def start(self, iter_round:int, batchsize:int, timestep_map = None, prepare_flag = False):
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
        if prepare_flag:
            self.fill_cache_data(5,256)
        logging.info(f"[LLMScheduler] Filling Completed. Start TEST!")
        self.process_prompt_batch()

    def start_test(self, iter_round:int, batchsize:int, loaded_datas = None,hack_option = None):
        test_user_counter = {}
        test_item_counter = {}
        if loaded_datas != None:
            load_data_user = loaded_datas['user']
            load_item_user = loaded_datas['item']
            timestep_map = loaded_datas['time_step_map']
            user_item_data = loaded_datas['user_candidate_dict']
        if not self.prompt_generator:
            print("[LLMScheduler] Error: prompt_generator is NONE!")
            return
        for timestep in range(iter_round):
            if loaded_datas != None:
                input_prompt_list = self.prompt_generator.generate_time_series_without_dataloader(batchsize,timestep,timestep_map,user_item_data,load_data_user,load_item_user)
            else:
                input_prompt_list = self.prompt_generator.generate(batchsize)
            self.add_prompt_list(input_prompt_list)
            # NOTE 平时不用运行，看商品数量能不能填满
            # for i in input_prompt_list:
            #     test_user_counter[i.user_id] = 0
            #     for item in i.items:
            #         test_item_counter[item.item_id] = 0
        # self.process_prompt()
        # logging.info(f"[LLMScheduler] Test User Num: {len(test_user_counter)} Test Item Num: {len(test_item_counter)}")
        self.process_prompt_batch(hack_option=hack_option)
        
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

    def fill_cache_data(self, iter_round:int, batchsize:int):
        logging.info(f"[LLMScheduler] Filling Cache Data...")
        prompt_list = []
        if not self.prompt_generator:
            print("[LLMScheduler] Error: prompt_generator is NONE!")
            return
        for _ in range(iter_round):
            input_prompt_list = self.prompt_generator.generate(batchsize)
            prompt_list.extend(input_prompt_list)
        cnt = 0
        working_prompt_list = []
        batch_size = self.num_workers * self.batchsize  # 每批次的大小
        total_prompts = len(prompt_list)
        for prompt in prompt_list:
            self._id_counter += 1
            prompt.task_id = self._id_counter
            working_prompt_list.append(prompt)
            cnt += 1
            if cnt % batch_size == 0:
                self._send_prompt_batch(working_prompt_list,None,True)
                working_prompt_list = []
        # 处理尾巴部分（未整除的 prompts）
        remaining = total_prompts % batch_size  # 剩余未整除的数量
        if remaining > 0:  # 检查是否有剩余
            self._send_prompt_batch(self.prompt_list[-remaining:],None,True)

    def schedule_strategy(self):
        pass

def schedule_order(prompt: InputPrompt,ans_dict = None, hook_option = None) -> str:
    if hook_option != None:
        return hook_option
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

def schedule_order_budget(prompt: InputPrompt, compute_budget) -> str:
    user_compute_tokens = prompt.user_history_tokens + prompt.miss_item_tokens
    item_compute_tokens = prompt.item_tokens + prompt.miss_user_history_tokens
    # 假设 item first 情况下 user his+item miss 超 budget 变 用户first 
    if user_compute_tokens >= compute_budget:
        return "User History First"
    elif item_compute_tokens >= compute_budget:
        logging.warning(f"[LLMScheduler] Item First compute tokens {item_compute_tokens} exceed budget {compute_budget}, still keep item first")
        return "Item First"
    # *其他*情况保持item first
    else:
        return "Item First"
    # 如果两者不满足（即变user还是超budget）报错
    
def schedule_order_compete(prompt: InputPrompt) -> str:
    user_tokens = prompt.user_history_tokens
    item_tokens = sum([item.token_count for item in prompt.items])
    # print(f"user {user_tokens}, item {item_tokens}")
    if user_tokens >= item_tokens:
        return "User History First"
    else:
        return "Item First"

def dict_to_taskinfo(task_dict):
    task = TaskInfo_pb2.TaskInfo()
    ParseDict(task_dict, task)
    return task