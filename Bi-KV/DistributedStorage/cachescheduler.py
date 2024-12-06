"cachescheduler.py"
import torch.distributed.rpc as rpc
from threading import Lock, Thread
from DistributedStorage.kvcache import KVCache
import time
from DistributedStorage.Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE

def call_receive_task_info(rref, task_info):
    """全局函数，用于调用 RRef 引用的 receive_task_info 方法"""
    return rref.rpc_sync().receive_task_info(task_info)

class CacheScheduler:
    def __init__(self, rank, world_size):
        """初始化调度器"""
        print("[CacheScheduler] 初始化调度器")
        self.rank = rank
        self.request_table = {}  # 用字典来存储请求表
        self.cpu_state_table = {rank: {'status': 'idle'} for rank in range(1, world_size)}  # 修改为 CPU 状态
        self.lock = Lock()
        self.kvcache_ref = []
        for i in range(0,world_size):
            print(f"[CacheScheduler] 创建远程实例 kvcache {i}")
            self.kvcache_ref.append(rpc.remote(f"kvcache{i}", KVCache, args=(i,)))  # 创建远程实例
    
    def add_requests(self, requests):
        """一次性添加多个请求到请求表"""
        for request in requests:
            request_id, send_cpu, recv_cpu = request
            self.add_request(request_id, send_cpu, recv_cpu)

    def add_request(self, request_id, send_cpu, recv_cpu):
        """添加单个请求到请求表"""
        print(f"[CacheScheduler] 添加请求：请求ID={request_id}, 发送CPU={send_cpu}, 接收CPU={recv_cpu}")
        self.request_table[request_id] = {'send_cpu': send_cpu, 'recv_cpu': recv_cpu, 'executing': False}

    def process_requests(self):
        """处理所有请求"""
        print("[CacheScheduler] 开始处理请求")
        while self.request_table:
            executable_requests = []

            # 遍历请求表中的所有请求
            for request_id, req in list(self.request_table.items()):
                send_cpu, recv_cpu, executing = req['send_cpu'], req['recv_cpu'], req['executing']
                if not executing and self.cpu_state_table.get(send_cpu, {}).get('status') == 'idle' and self.cpu_state_table.get(recv_cpu, {}).get('status') == 'idle':
                    with self.lock:
                        # 标记 CPU 状态为 busy
                        self.cpu_state_table[send_cpu]['status'] = 'sending'
                        self.cpu_state_table[recv_cpu]['status'] = 'receiving'
                        # 更新请求状态
                        self.request_table[request_id]['executing'] = True
                    # 添加到可执行请求列表
                    executable_requests.append(request_id)

            if not executable_requests:
                # 如果没有可执行的请求，稍等片刻再检查
                time.sleep(0.1)
                continue

            # 按请求ID排序，优先处理较早的请求
            executable_requests.sort()

            # 并发执行可执行请求
            threads = []
            for request_id in executable_requests:
                req = self.request_table[request_id]
                send_cpu, recv_cpu = req['send_cpu'], req['recv_cpu']
                thread = Thread(target=self._execute_request, args=(request_id, send_cpu, recv_cpu))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

        print("[CacheScheduler] 所有请求处理完成")
        return

    def _execute_request(self, request_id, send_cpu, recv_cpu):
        """执行单个请求"""
        print(f"[CacheScheduler] 执行请求 {request_id} - CPU {send_cpu} -> CPU {recv_cpu}")
        
        # 发送发送任务到发送CPU（通过 RPC）
        task_info_send = [SIGNAL_SEND, request_id, send_cpu, recv_cpu]
        future_send = rpc.rpc_async(self.kvcache_ref[send_cpu].owner(), call_receive_task_info, args=(self.kvcache_ref[send_cpu], task_info_send))

        # 发送接收任务到接收CPU（通过 RPC）
        task_info_recv = [SIGNAL_RECV, request_id, send_cpu, recv_cpu]
        future_recv = rpc.rpc_async(self.kvcache_ref[recv_cpu].owner(), call_receive_task_info, args=(self.kvcache_ref[recv_cpu], task_info_recv))
        
        future_send.wait()  # 确保任务完成
        confirmation_msg = future_recv.wait()
        if confirmation_msg == request_id:
            print(f"[CacheScheduler] 请求 {request_id} 完成 - CPU {send_cpu} -> CPU {recv_cpu}")

        with self.lock:
            del self.request_table[request_id]  # 删除请求
            self.cpu_state_table[send_cpu]['status'] = 'idle'
            self.cpu_state_table[recv_cpu]['status'] = 'idle'

    def send_terminate_signal(self):
        """通过 RPC 发送终止信号给所有 KVCache"""
        print("[CacheScheduler] 发送终止信号给所有 KVCache")
        for cpu_rank in self.cpu_state_table.keys():
            # 使用 RPC 发送终止信号
            rpc.rpc_sync(f"worker{cpu_rank}", KVCache.terminate)
        print("[CacheScheduler] 终止信号已发送")
        rpc.shutdown()
