from typing import Dict, List
import time
from queue import Queue

import torch.distributed.rpc as rpc
from threading import Lock, Thread
from DistributedStorage.kvcache import KVCache
from DistributedStorage.Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
from Remote.remote_call import call_remote_method
from rpc_def import KVCACHE_offset,WORKER_offset

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
    
    def add_requests(self, requests:List[Dict]):
        for request in requests:
            self.add_request(request)

    def add_request(self, task_info:Dict):
        request_id = task_info["request_id"]
        recv_worker = task_info["recv_worker"]
        print(f"[CacheCoordinator] 添加请求：请求ID={request_id}, 接收Rank={recv_worker+WORKER_offset}")
        # TODO 需要补全cache miss逻辑，且用strategy庖代
        task_info["send_worker"] = self.strategy(request_id+task_info['id'])
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
                req = self.request_table.get_nowait()
                send_worker, recv_worker, executing = req['send_worker'], req['recv_worker'], req['executing']
                if not executing and self.cpu_state_table[send_worker]['status'] == 'idle' and self.cpu_state_table[recv_worker]['status'] == 'idle':
                    with self.lock:
                        self.cpu_state_table[send_worker]['status'] = 'sending'
                        self.cpu_state_table[recv_worker]['status'] = 'receiving'
                        req['executing'] = True
                    self.finished_counter_table[req['request_id']] = 0
                    self.finished_flag_table[req['request_id']] = False
                    executable_requests.append(req)
                    has_excuted = True
                else:
                    # 无法执行则加入无法执行list，后续重新入队
                    unexecutable_requests.append(req)
            for req in unexecutable_requests:
                self.request_table.put_nowait(req)
            if not executable_requests and not has_excuted:
                time.sleep(0.1)
                # print(f"[CacheCoordinator] Empty executable_requests. Waiting...")
                continue

            executable_requests.sort(key=lambda x: x['request_id'])
            threads = []
            for req in executable_requests:
                # print(f"[CacheCoordinator] About to execute {req}")
                thread = Thread(target=self._execute_request, args=(req,))
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
                time.sleep(10)
                continue

        print("[CacheCoordinator] 所有请求处理完成")

    def _execute_request(self, req:Dict):
        request_id, send_worker, recv_worker = req['request_id'], req['send_worker'], req['recv_worker']
        self.finished_counter_table[request_id] -= 1
        print(f"[CacheCoordinator] 执行请求 {request_id} - Rank {send_worker+KVCACHE_offset} -> Rank {recv_worker+WORKER_offset}")
        req["task_type"] = SIGNAL_SEND
        future_send = rpc.rpc_async(
            self.kvcache_ref[send_worker].owner(),
            call_remote_method, 
            args=(KVCache.receive_task_info,
                self.kvcache_ref[send_worker], 
                req,self.worker_ref[recv_worker]))

        # task_info_recv = [SIGNAL_RECV, request_id, send_cpu, recv_cpu]
        # future_recv = rpc.rpc_async(self.kvcache_ref[recv_cpu].owner(), _call_remote_method, args=(KVCache.receive_task_info,self.kvcache_ref[recv_cpu], task_info_recv,self.worker_ref[recv_cpu]))
        
        confirmation_msg = future_send.wait()
        # confirmation_msg = future_recv.wait()
        if confirmation_msg == request_id:
            print(f"[CacheCoordinator] 请求 {request_id} 完成 - Rank {send_worker+KVCACHE_offset} -> Rank {recv_worker+WORKER_offset}")

        with self.lock:
            # del self.request_table[request_id]
            self.cpu_state_table[send_worker]['status'] = 'idle'
            self.cpu_state_table[recv_worker]['status'] = 'idle'
        self.finished_counter_table[request_id] += 1
        self.finished_flag_table[request_id] = True

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
    
    def poll(self,request_id:int):
        # 一组task_info_list应该用的是同一个request_id
        res_counter = self.finished_counter_table.get(request_id,-1)
        res_flag = self.finished_flag_table.get(request_id, False)
        if res_counter == 0 and res_flag:
            return True
        return False

    def stop_process(self):
        self.process_flag = False

    def set_stop_limit(self, stop_limit:int):
        self.stop_limit = stop_limit