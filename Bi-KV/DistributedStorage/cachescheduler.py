import torch.distributed.rpc as rpc
from threading import Lock, Thread
from DistributedStorage.kvcache import KVCache
import time
from DistributedStorage.Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
from Reomte.remote_call import _call_remote_method

class CacheScheduler:
    def __init__(self, rank, kvcache_num):
        """初始化调度器"""
        print("[CacheScheduler] 初始化调度器")
        self.kvcache_num = kvcache_num
        self.rank = rank
        self.request_table = {}
        # cpu_state_table记录每个kvcache的状态(0-based)
        self.cpu_state_table = {i: {'status': 'idle'} for i in range(self.kvcache_num)}
        self.lock = Lock()
        self.kvcache_ref = []
        for i in range(self.kvcache_num):
            print(f"[CacheScheduler] 创建远程实例 kvcache {i}")
            self.kvcache_ref.append(rpc.remote(f"kvcache{i}", KVCache, args=(i,)))  # 创建远程实例
    
    def add_requests(self, requests):
        for request in requests:
            request_id, send_cpu, recv_cpu = request
            self.add_request(request_id, send_cpu, recv_cpu)

    def add_request(self, request_id, send_cpu, recv_cpu):
        print(f"[CacheScheduler] 添加请求：请求ID={request_id}, 发送CPU={send_cpu}, 接收CPU={recv_cpu}")
        self.request_table[request_id] = {'send_cpu': send_cpu, 'recv_cpu': recv_cpu, 'executing': False}

    def process_requests(self):
        print("[CacheScheduler] 开始处理请求")
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

        print("[CacheScheduler] 所有请求处理完成")
        return

    def _execute_request(self, request_id, send_cpu, recv_cpu):
        print(f"[CacheScheduler] 执行请求 {request_id} - CPU {send_cpu} -> CPU {recv_cpu}")
        task_info_send = [SIGNAL_SEND, request_id, send_cpu, recv_cpu]
        future_send = rpc.rpc_async(self.kvcache_ref[send_cpu].owner(),_call_remote_method, args=(KVCache.receive_task_info,self.kvcache_ref[send_cpu], task_info_send))

        task_info_recv = [SIGNAL_RECV, request_id, send_cpu, recv_cpu]
        future_recv = rpc.rpc_async(self.kvcache_ref[recv_cpu].owner(), _call_remote_method, args=(KVCache.receive_task_info,self.kvcache_ref[recv_cpu], task_info_recv))
        
        future_send.wait()
        confirmation_msg = future_recv.wait()
        if confirmation_msg == request_id:
            print(f"[CacheScheduler] 请求 {request_id} 完成 - CPU {send_cpu} -> CPU {recv_cpu}")

        with self.lock:
            del self.request_table[request_id]
            self.cpu_state_table[send_cpu]['status'] = 'idle'
            self.cpu_state_table[recv_cpu]['status'] = 'idle'

    def send_terminate_signal(self):
        print("[CacheScheduler] 发送终止信号给所有 KVCache")
        for cpu_rank in range(self.kvcache_num):
            rpc.rpc_async(self.kvcache_ref[cpu_rank].owner(), _call_remote_method, args=(KVCache.terminate,self.kvcache_ref[cpu_rank],))
        print("[CacheScheduler] 终止信号已发送")
        return
