from typing import Tuple,Dict, List
import time
from queue import Queue

import torch.distributed.rpc as rpc
from threading import Lock, Thread
from DistributedStorage.kvcache import KVCache
from DistributedStorage.Storage import LRUCache
from DistributedStorage.PageManager import MultiPageManager
from DistributedStorage.Signals import SIGNAL_CHECK, SIGNAL_SEND, SIGNAL_RECV, CACHE_MISS, CACHE_HIT,SIGNAL_SKIP
from Remote.remote_call import call_remote_method
from rpc_def import KVCACHE_offset,WORKER_offset,PROCESS_TYPES, WORKER_NUM, KVCACHE_NUM, get_process_info
from config import *

class CacheCoordinator:
    def __init__(self, rank, kvcache_num):
        """初始化调度器"""
        print("[CacheCoordinator] 初始化调度器")
        self.kvcache_num = kvcache_num
        self.rank = rank
        self.worker_ref = None
        self.request_table = Queue()
        self.finished_counter_table = {}
        self.finished_flag_table = {}
        self.process_flag = False
        # cpu_state_table记录每个kvcache的状态(0-based)
        self.cpu_state_table = {i: {'status': 'idle'} for i in range(self.kvcache_num)}
        self.lock = Lock()
        self.kvcache_ref = []
        self.stop_limit = 10000
        self.cache = 5000
        self.page_size = 50
        for i in range(self.kvcache_num):
            cache_rank= 2*i+3 
            proc_type, proc_index = get_process_info(cache_rank)
            rpc_info = rpc.get_worker_info(f"{proc_type}{proc_index}")
            print(f"[CacheCoordinator] 创建远程实例 KVCache {i}")
            self.kvcache_ref.append(rpc.remote(rpc_info, KVCache, args=(cache_rank,self.cache,self.page_size,)))  # 创建远程实例
        self.page_miss_dict = {}

        # 测试MultiPageManager
        self.page_manager = MultiPageManager(self.cache, self.page_size, self.kvcache_num)

    
    def add_requests(self, requests:List[Dict]):
        for request in requests:
            self.add_request(request)

    def add_request(self, task_info:Dict):
        request_id = task_info["request_id"]
        infer_worker = task_info["infer_worker"]
        if DEBUG:
            print(f"[CacheCoordinator] 添加请求：请求ID={request_id}, 接收Wroker{infer_worker}Rank={2*infer_worker+2}")
        # TODO 需要补全cache miss逻辑，且用strategy庖代
        task_info["cache_worker"] = self.strategy(request_id+task_info['id'])
        task_info["executing"] = False
        self.request_table.put(task_info)
        #print(f"task_info{task_info}")


    def process_requests(self):
        print("[CacheCoordinator] 开始处理请求")
        idle_time_counter = 0
        has_excuted = False
        while self.process_flag:
            # executable_requests = []
            executable_requests = {}
            
            # time0 = time.time()

            while not self.request_table.empty():
                idle_time_counter = 0
                task_info = self.request_table.get_nowait()
                req_id = task_info['request_id']
                if True: 
                    if task_info['task_type'] == SIGNAL_CHECK:
                        if self.page_miss_dict.get(req_id)==None:
                            self.page_miss_dict[req_id] = {}
                        access_res = self.page_manager.access_item(task_info['id'])
                        if access_res[0] == None:
                            # if DEBUG:
                            # print(f"[CacheCoordinator] Cache Miss! id = {task_info['id']}")
                            self.page_miss_dict[req_id][task_info['id']] = CACHE_MISS
                        else:
                            # cache hit
                            cache_worker = access_res[0]
                            pages_list = access_res[1]
                            self.page_miss_dict[req_id][task_info['id']] = CACHE_HIT
                            task_info['task_type'] = SIGNAL_SEND
                            task_info['cache_worker'] = cache_worker
                            task_info['cache_pages_list'] = pages_list

                    if task_info['task_type'] == SIGNAL_RECV:
                        load_res = self.page_manager.load_item(task_info['id'], task_info['token_num'])
                        cache_worker = load_res[0]
                        pages_list = load_res[1]
                        task_info['cache_worker'] = cache_worker
                        task_info['cache_pages_list'] = pages_list

                    cache_worker = task_info['cache_worker']

                    # 初始化finished_counter_table
                    if self.finished_counter_table.get(task_info['request_id']) == None:
                        self.finished_counter_table[task_info['request_id']] = 0
                    # 初始化finished_flag_table
                    if executable_requests.get(cache_worker) == None:
                        executable_requests[cache_worker] = [task_info]
                    else:
                        executable_requests[cache_worker].append(task_info)
                    # 改变状态为has_excuted
                    has_excuted = True
            
            # 若executable_requests为空且从未执行过，则继续等待直到有请求
            if not executable_requests and not has_excuted:
                continue

            threads = []
            # time1 = time.time()
            # count = 0
            cache_future = []
            for cache_worker in executable_requests:
                future = self._execute_request_batch(executable_requests[cache_worker], cache_worker) 
                cache_future.append(future)
                # count = 0
                # batched_request = []
                # for request in executable_requests[cache_worker]:
                #     batched_request.append(request)                    
                #     count += 1
                #     if count % 512 == 0:
                #         self._execute_request_batch(batched_request, cache_worker) 
                #         batched_request = []
                # if len(batched_request) > 0:
                #     self._execute_request_batch(batched_request, cache_worker) 

            for future in cache_future:
                # confirmation_msg是一个字典，key是request_id，value是完成的task数量
                confirmation_msg = future.wait()
                if len(confirmation_msg) > 0:
                    if DEBUG:
                        request_id=task_info['request_id']
                        print(f"[CacheCoordinator] 请求 {request_id} 完成 - Rank {2*cache_worker+3}")
                    for request_id in confirmation_msg:
                        self.finished_counter_table[request_id] += confirmation_msg[request_id]
                
            
            if idle_time_counter>self.stop_limit and self.request_table.empty():
                print(f"[CacheCoordinator] Empty request table. B R E A K")
                self.send_terminate_signal()
                break
            # time2 = time.time()
            if self.request_table.empty():
                # print(f"[CacheCoordinator] Empty request table. Waiting...({idle_time_counter})")
                idle_time_counter+=1
                time.sleep(0.0005)
                continue
            # print(f"execute times cost {(time.time()-time2)}s, {(time2-time1)}s, {(time1-time0)}s")

        print("[CacheCoordinator] 所有请求处理完成")

    def _execute_request(self, req:Dict):
        request_id, cache_worker, infer_worker = req['request_id'], req['cache_worker'], req['infer_worker']
        if DEBUG:
            print(f"[CacheCoordinator] 执行请求ID= {request_id} - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
        # 若这里仍然是Check，则不执行
        if req['task_type'] == SIGNAL_CHECK or req['task_type'] == SIGNAL_SKIP:
            confirmation_msg = request_id
        else:
            future_send = rpc.rpc_async(
                self.kvcache_ref[cache_worker].owner(),
                call_remote_method, 
                args=(KVCache.receive_task_info,
                    self.kvcache_ref[cache_worker], 
                    req,self.worker_ref[infer_worker]))

            confirmation_msg = future_send.wait()
        if confirmation_msg == request_id:
            if DEBUG:
                print(f"[CacheCoordinator] 请求 {request_id} 完成 - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
            self.finished_counter_table[request_id] += 1
            self.finished_flag_table[request_id] = True

    def _execute_request_batch(self, req_list:List[Dict], cache_worker):
        if DEBUG:
            request_id = req_list[0]['request_id']
            infer_worker = req_list[0]['infer_worker']
            print(f"[CacheCoordinator] 执行请求ID= {request_id} - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
        # TODO 若这里仍然是Check，则应该不执行
        future = rpc.rpc_async(
            self.kvcache_ref[cache_worker].owner(),
            call_remote_method, 
            args=(KVCache.receive_task_info_batch,
                self.kvcache_ref[cache_worker],self.worker_ref, 
                req_list))
        if DEBUG:
            print(f"[CacheCoordinator]finish _execute_request_batch")
        return future

            
    def send_terminate_signal(self):
        print("[CacheCoordinator] 发送终止信号给所有 KVCache")
        futures = []
        for cpu_rank in range(self.kvcache_num):
            print(f"[CacheCoordinator] Trying to terminate KVCache {cpu_rank}")
            fut = rpc.rpc_async(self.kvcache_ref[cpu_rank].owner(), 
                          call_remote_method, args=(KVCache.terminate,self.kvcache_ref[cpu_rank],))
            futures.append(fut)
        for fut in futures:
            fut.wait()
        print("[CacheCoordinator] 终止信号已发送")
        

    def set_workers_rref(self,workers_rref):
        self.worker_ref = workers_rref
        print(f"[CacheCoordinator] 已设置workers rref信息 长度为{len(self.worker_ref)}")
        self.process_flag = True
        self.process_requests()
    
    def strategy(self, req_id: int) -> int:
        return req_id % self.kvcache_num
    
    def poll(self,task_info_list:List[Dict])->Tuple[bool, Dict]:
        cache_miss_dict = {}
        # 一组task_info_list应该用的是同一个request_id
        request_id = task_info_list[0]['request_id']
        task_list_length = len(task_info_list)
        res_counter = self.finished_counter_table.get(request_id,-1)
        res_flag = self.finished_flag_table.get(request_id, False)
        # print(f"[CacheCoordinator] lru_miss_dict: {self.lru_miss_dict}")
        for i in task_info_list:
            # 如果得到None是不是得多做一些操作？
            if self.page_miss_dict.get(request_id)==None:
                # print(f"[CacheCoordinator] Error: request_id {request_id} not found in lru_miss_dict")
                continue
            if i['id'] == -1:
                continue
            # print(f"[CacheCoordinator] Polling id {i['id']}")
            cache_miss_dict[i['id']] = self.page_miss_dict[request_id].get(i['id'],-1)
        if res_counter == task_list_length and res_flag:
            return True, cache_miss_dict
        return False, cache_miss_dict
    
    def poll_batch(self,task_info_list:List[Dict])->Tuple[bool, Dict]:
        cache_miss_dict = {}
        request_to_task_num = {}
        for task_info in task_info_list:
            request_id = task_info['request_id']
            task_num = task_info['task_num'] # task_num代表一个req_id中的task数量
            if request_id not in request_to_task_num:
                request_to_task_num[request_id] = task_num
            else:
                if request_to_task_num[request_id] != task_num:
                    raise ValueError(f"Conflicting task_num for request id {request_id}")
        finish_count = 0

        unfinished_requests = set(request_to_task_num.keys())  # 未完成的 request_id 集合

        while finish_count != len(request_to_task_num):
            for request_id in list(unfinished_requests):  # 遍历未完成的 request_id
                task_num = request_to_task_num[request_id]
                res_counter = self.finished_counter_table.get(request_id, -1)
                
                # 判断任务是否完成
                if res_counter == task_num:
                    # self.lru_miss_dict[req_id][task_info['id']] = Cache状态 
                    cache_miss_dict[request_id] = self.page_miss_dict.get(request_id)
                    finish_count += 1
                    unfinished_requests.remove(request_id)  # 移除已完成的 request_id

        return cache_miss_dict

    def stop_process(self):
        self.process_flag = False

    def set_stop_limit(self, stop_limit:int):
        self.stop_limit = stop_limit

    def test_write(self, task_info:Dict):
        infer_worker = task_info['infer_worker']
        recv_cache_ref = self.kvcache_ref[infer_worker]
        # print(f"[CacheCoordinator] Trying send message to kvcache{infer_worker}")
        owner_cache_ref = recv_cache_ref.owner()
        rpc.rpc_sync(to=owner_cache_ref, func=call_remote_method, 
                            args=(KVCache.receive_data, recv_cache_ref, task_info))