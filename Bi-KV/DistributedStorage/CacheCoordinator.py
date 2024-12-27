from typing import Tuple,Dict, List
import time
from queue import Queue

from httpx import request
import torch.distributed.rpc as rpc
from threading import Lock, Thread
from DistributedStorage.kvcache import KVCache
from DistributedStorage.Storage import LRUCache
from DistributedStorage.Signals import SIGNAL_CHECK, SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
from Remote.remote_call import call_remote_method
from rpc_def import KVCACHE_offset,WORKER_offset
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
        self.stop_limit = 10
        for i in range(self.kvcache_num):
            print(f"[CacheCoordinator] 创建远程实例 kvcache {i}")
            self.kvcache_ref.append(rpc.remote(f"kvcache{i}", KVCache, args=(i,)))  # 创建远程实例
        self.lru_capacity = 1000
        self.lru = LRUCache(self.lru_capacity, self.kvcache_num)
        self.lru_miss_dict = {}
    
    def add_requests(self, requests:List[Dict]):
        for request in requests:
            self.add_request(request)

    def add_request(self, task_info:Dict):
        request_id = task_info["request_id"]
        infer_worker = task_info["infer_worker"]
        if DEBUG:
            print(f"[CacheCoordinator] 添加请求：请求ID={request_id}, 接收Rank={infer_worker+WORKER_offset}")
        # TODO 需要补全cache miss逻辑，且用strategy庖代
        task_info["cache_worker"] = self.strategy(request_id+task_info['id'])
        task_info["executing"] = False
        self.request_table.put(task_info)

    def process_requests(self):
        print("[CacheCoordinator] 开始处理请求")
        idle_time_counter = 0
        has_excuted = False
        while self.process_flag:
            executable_requests = []
            unexecutable_requests = []
            while not self.request_table.empty():
                idle_time_counter = 0
                task_info = self.request_table.get_nowait()
                # print(f"[CacheCoordinator] Processing request type: {req['task_type']}")
                req_id = task_info['request_id']
                cache_worker, infer_worker, executing = task_info['cache_worker'], task_info['infer_worker'], task_info['executing']
                if not executing and self.cpu_state_table[cache_worker]['status'] == 'idle' and self.cpu_state_table[infer_worker]['status'] == 'idle':
                    with self.lock:
                        self.cpu_state_table[cache_worker]['status'] = 'sending'
                        self.cpu_state_table[infer_worker]['status'] = 'receiving'
                        task_info['executing'] = True
                    if self.finished_counter_table.get(task_info['request_id'])==None:
                        self.finished_counter_table[task_info['request_id']] = 0
                    self.finished_flag_table[task_info['request_id']] = False
                    executable_requests.append(task_info)
                    if task_info['task_type'] == SIGNAL_CHECK:
                        if self.lru_miss_dict.get(req_id)==None:
                            self.lru_miss_dict[req_id] = {}
                        if self.lru.get(task_info)==None:
                            if DEBUG:
                                print(f"[CacheCoordinator] Cache Miss! id = {task_info['id']}")
                            # 改为在worker写入cache后再put
                            # self.lru.put(req)
                            self.lru_miss_dict[req_id][task_info['id']] = 0
                        else:
                            # cache hit 调取数据
                            self.lru_miss_dict[req_id][task_info['id']] = 1
                            task_info['task_type'] = SIGNAL_SEND
                    if task_info['task_type'] == SIGNAL_RECV:
                        self.lru.put(task_info)
                    has_excuted = True
                else:
                    # 无法执行则加入无法执行list，后续重新入队
                    unexecutable_requests.append(task_info)
            for task_info in unexecutable_requests:
                self.request_table.put_nowait(task_info)
            if not executable_requests and not has_excuted:
                time.sleep(0.1)
                # print(f"[CacheCoordinator] Empty executable_requests. Waiting...")
                continue

            executable_requests.sort(key=lambda x: x['request_id'])
            threads = []
            for task_info in executable_requests:
                # print(f"[CacheCoordinator] About to execute {req}")
                thread = Thread(target=self._execute_request, args=(task_info,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            if idle_time_counter>self.stop_limit and self.request_table.empty():
                print(f"[CacheCoordinator] Empty request table. B R E A K")
                self.send_terminate_signal()
                break

            if self.request_table.empty():
                print(f"[CacheCoordinator] Empty request table. Waiting...({idle_time_counter})")
                idle_time_counter+=1
                time.sleep(3)
                continue

        print("[CacheCoordinator] 所有请求处理完成")

    def _execute_request(self, req:Dict):
        request_id, cache_worker, infer_worker = req['request_id'], req['cache_worker'], req['infer_worker']
        if DEBUG:
            print(f"[CacheCoordinator] 执行请求 {request_id} - Rank {cache_worker+KVCACHE_offset} -> Rank {infer_worker+WORKER_offset}")
        # 若这里仍然是Check，则不执行
        if req['task_type'] == SIGNAL_CHECK or req['task_type'] == SIGNAL_ACK:
            confirmation_msg = request_id
        else:
            future_send = rpc.rpc_async(
                self.kvcache_ref[cache_worker].owner(),
                call_remote_method, 
                args=(KVCache.receive_task_info,
                    self.kvcache_ref[cache_worker], 
                    req,self.worker_ref[infer_worker]))

            # task_info_recv = [SIGNAL_RECV, request_id, send_cpu, recv_cpu]
            # future_recv = rpc.rpc_async(self.kvcache_ref[recv_cpu].owner(), _call_remote_method, args=(KVCache.receive_task_info,self.kvcache_ref[recv_cpu], task_info_recv,self.worker_ref[recv_cpu]))
            
            confirmation_msg = future_send.wait()
            # confirmation_msg = future_recv.wait()
        if confirmation_msg == request_id:
            if DEBUG:
                print(f"[CacheCoordinator] 请求 {request_id} 完成 - Rank {cache_worker+KVCACHE_offset} -> Rank {infer_worker+WORKER_offset}")
            self.finished_counter_table[request_id] += 1
            self.finished_flag_table[request_id] = True

        with self.lock:
            # del self.request_table[request_id]
            self.cpu_state_table[cache_worker]['status'] = 'idle'
            self.cpu_state_table[infer_worker]['status'] = 'idle'

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
        # print(f"[CacheCoordinator] counter: {res_counter} length:{task_list_length}")
        for i in task_info_list:
            # 如果得到None是不是得多做一些操作？
            if self.lru_miss_dict.get(request_id)==None:
                continue
            if i['id'] == -1:
                continue
            cache_miss_dict[i['id']] = self.lru_miss_dict[request_id].get(i['id'],-1)
        if res_counter == task_list_length and res_flag:
            return True, cache_miss_dict
        return False, cache_miss_dict
    
    def batch_poll(self,task_info_batch:List[List[Dict]]):
        res = {}
        for task_info_list in task_info_batch:
            request_id = task_info_list[0]['request_id']
            res[request_id] = self.poll(task_info_list)
        return res

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