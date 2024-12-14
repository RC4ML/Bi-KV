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
        # TODO request_table转为队列
        self.request_table = Queue()
        # self.request_table = {}
        # cpu_state_table记录每个kvcache的状态(0-based)
        self.cpu_state_table = {i: {'status': 'idle'} for i in range(self.kvcache_num)}
        self.lock = Lock()
        self.kvcache_ref = []
        for i in range(self.kvcache_num):
            print(f"[CacheCoordinator] 创建远程实例 kvcache {i}")
            self.kvcache_ref.append(rpc.remote(f"kvcache{i}", KVCache, args=(i,)))  # 创建远程实例
    
    def add_requests(self, requests):
        for request in requests:
            request_id, send_cpu, recv_cpu = request
            self.add_request(request_id, send_cpu, recv_cpu)

    def add_request(self, request_id, send_cpu, recv_cpu):
        print(f"[CacheCoordinator] 添加请求：请求ID={request_id}, 发送Rank={send_cpu+KVCACHE_offset}, 接收Rank={recv_cpu+WORKER_offset}")
        # self.request_table[request_id] = {'send_cpu': send_cpu, 'recv_cpu': recv_cpu, 'executing': False}
        self.request_table.put({'request_id':request_id,'send_cpu': send_cpu, 'recv_cpu': recv_cpu, 'executing': False})

    def process_requests(self):
        print("[CacheCoordinator] 开始处理请求")
        while not self.request_table.empty():
            executable_requests = []
            unexecutable_requests = []
            while not self.request_table.empty():
                req = self.request_table.get_nowait()
                request_id = req['request_id']
                send_cpu, recv_cpu, executing = req['send_cpu'], req['recv_cpu'], req['executing']
                if not executing and self.cpu_state_table[send_cpu]['status'] == 'idle' and self.cpu_state_table[recv_cpu]['status'] == 'idle':
                    with self.lock:
                        self.cpu_state_table[send_cpu]['status'] = 'sending'
                        self.cpu_state_table[recv_cpu]['status'] = 'receiving'
                        req['executing'] = True
                    executable_requests.append(req)
                else:
                    # 无法执行则加入不无法执行list，后续重新入队
                    unexecutable_requests.append(req)
            for req in unexecutable_requests:
                self.request_table.put_nowait(req)
            if not executable_requests:
                time.sleep(0.1)
                continue

            executable_requests.sort(key=lambda x: x['request_id'])
            threads = []
            for req in executable_requests:
                request_id = req['request_id']
                send_cpu, recv_cpu = req['send_cpu'], req['recv_cpu']
                thread = Thread(target=self._execute_request, args=(request_id, send_cpu, recv_cpu))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        print("[CacheCoordinator] 所有请求处理完成")

    def process_requests_old(self):
        print("[CacheCoordinator] 开始处理请求")
        while self.request_table:
            executable_requests = []
            for request_id, req in list(self.request_table.items()):
                send_cpu, recv_cpu, executing = req['send_cpu'], req['recv_cpu'], req['executing']
                if not executing and self.cpu_state_table[send_cpu]['status'] == 'idle' and self.cpu_state_table[recv_cpu]['status'] == 'idle':
                    with self.lock:
                        self.cpu_state_table[send_cpu]['status'] = 'sending'
                        self.cpu_state_table[recv_cpu]['status'] = 'receiving'
                        self.request_table[request_id]['executing'] = True
                    executable_requests.append(request_id)

            if not executable_requests:
                time.sleep(0.1)
                continue

            executable_requests.sort()
            threads = []
            for request_id in executable_requests:
                req = self.request_table[request_id]
                send_cpu, recv_cpu = req['send_cpu'], req['recv_cpu']
                thread = Thread(target=self._execute_request, args=(request_id, send_cpu, recv_cpu))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        print("[CacheCoordinator] 所有请求处理完成")
        return

    def _execute_request(self, request_id, send_cpu, recv_cpu):
        print(f"[CacheCoordinator] 执行请求 {request_id} - Rank {send_cpu+KVCACHE_offset} -> Rank {recv_cpu+WORKER_offset}")
        task_info_send = [SIGNAL_SEND, request_id, send_cpu, recv_cpu]
        future_send = rpc.rpc_async(
            self.kvcache_ref[send_cpu].owner(),
            call_remote_method, 
            args=(KVCache.receive_task_info,
                self.kvcache_ref[send_cpu], 
                task_info_send,self.worker_ref[recv_cpu]))

        # task_info_recv = [SIGNAL_RECV, request_id, send_cpu, recv_cpu]
        # future_recv = rpc.rpc_async(self.kvcache_ref[recv_cpu].owner(), _call_remote_method, args=(KVCache.receive_task_info,self.kvcache_ref[recv_cpu], task_info_recv,self.worker_ref[recv_cpu]))
        
        confirmation_msg = future_send.wait()
        # confirmation_msg = future_recv.wait()
        if confirmation_msg == request_id:
            print(f"[CacheCoordinator] 请求 {request_id} 完成 - Rank {send_cpu+KVCACHE_offset} -> Rank {recv_cpu+WORKER_offset}")

        with self.lock:
            # del self.request_table[request_id]
            self.cpu_state_table[send_cpu]['status'] = 'idle'
            self.cpu_state_table[recv_cpu]['status'] = 'idle'

    def send_terminate_signal(self):
        print("[CacheCoordinator] 发送终止信号给所有 KVCache")
        for cpu_rank in range(self.kvcache_num):
            rpc.rpc_async(self.kvcache_ref[cpu_rank].owner(), 
                          call_remote_method, args=(KVCache.terminate,self.kvcache_ref[cpu_rank],))
        print("[CacheCoordinator] 终止信号已发送")
        return

    def set_workers_rref(self,workers_rref):
        self.worker_ref = workers_rref
        print(f"[CacheCoordinator] 已设置worker rref信息 长度为{len(self.worker_ref)}")
        # send_worker_ref = self.worker_ref[0]
        # owner_worker_ref = send_worker_ref.owner()  
        # from Worker.Worker import Worker
        # future = rpc.rpc_sync(to=owner_worker_ref, func=_call_remote_method, args=(Worker.receive_task_info, send_worker_ref, "测试消息传递"))