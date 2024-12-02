import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from threading import Lock, Thread
from Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
import time
from kvcache import KVCache

def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)
class CacheScheduler:
    def __init__(self, world_size):
        """初始化调度器"""
        print("[CacheScheduler] 初始化调度器")
        self.request_table = {}  # 用字典来存储请求表
        self.gpu_state_table = {rank: {'status': 'idle'} for rank in range(1, world_size)}
        self.lock = Lock()
        # kvcache's rref
        self.cordinator_ref =rpc.RRef(self)
        self.cache_ref =[]
        for cache_rank in range (1,world_size):
            cache_info=rpc.get_worker_info(f"KVCache{cache_rank}")
            print(f"worker_info{cache_info}")
            self.cache_ref.append(rpc.remote(to=cache_info,func=KVCache,args=(cache_rank,)))
    def add_requests(self, requests):
        """一次性添加多个请求到请求表"""
        for request in requests:
            request_id, send_gpu, recv_gpu = request
            self.add_request(request_id, send_gpu, recv_gpu)

    def add_request(self, request_id, send_gpu, recv_gpu):
        """添加单个请求到请求表"""
        print(f"[CacheScheduler] 添加请求：请求ID={request_id}, 发送GPU={send_gpu}, 接收GPU={recv_gpu}")
        self.request_table[request_id] = {'send_gpu': send_gpu, 'recv_gpu': recv_gpu, 'executing': False}

    def process_requests(self):
        """处理所有请求"""
        print("[CacheScheduler] 开始处理请求")
        while self.request_table:
            executable_requests = []

            # 遍历请求表中的所有请求
            for request_id, req in list(self.request_table.items()):
                send_gpu, recv_gpu, executing = req['send_gpu'], req['recv_gpu'], req['executing']
                if not executing and self.gpu_state_table.get(send_gpu, {}).get('status') == 'idle' and self.gpu_state_table.get(recv_gpu, {}).get('status') == 'idle':
                    with self.lock:
                        # 标记 GPU 状态为 busy
                        self.gpu_state_table[send_gpu]['status'] = 'sending'
                        self.gpu_state_table[recv_gpu]['status'] = 'receiving'
                        # 更新请求状态
                        self.request_table[request_id]['executing'] = True
                    # 添加到可执行请求列表
                    executable_requests.append(request_id)

            # if not executable_requests:
            #     # 如果没有可执行的请求，稍等片刻再检查
            #     time.sleep(0.1)
            #     continue

            # 按请求ID排序，优先处理较早的请求
            executable_requests.sort()

            # 并发执行可执行请求
            threads = []
            for request_id in executable_requests:
                req = self.request_table[request_id]
                send_gpu, recv_gpu = req['send_gpu'], req['recv_gpu']
                thread = Thread(target=self._execute_request, args=(request_id, send_gpu, recv_gpu))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()
            break

        print("[CacheScheduler] 所有请求处理完成")
        return

    def _execute_request(self, request_id, send_gpu, recv_gpu):
        """执行单个请求"""
        print(f"[CacheScheduler] 执行请求 {request_id} - GPU {send_gpu} -> GPU {recv_gpu}")

        # 发送发送任务到发送GPU（通过 RPC）
        task_info_send = [SIGNAL_SEND, request_id, send_gpu, recv_gpu]
        send_cache_ref=self.cache_ref[send_gpu-1]
        # 获取发送 GPU 的拥有者，并在拥有者上调用 local_value
        owner_worker_ref = send_cache_ref.owner()  # 获取 RRef 的拥有者
        rpc.rpc_sync(to=owner_worker_ref, func=_call_method, args=(KVCache.receive_task_info,send_cache_ref, task_info_send),timeout=1)
        print("传输结束")

    def send_terminate_signal(self):
        """通过 RPC 发送终止信号给所有 KVCache"""
        print("[CacheScheduler] 发送终止信号给所有 KVCache")
        for send_cache_ref in self.cache_ref:
            # 使用 RPC 发送终止信号
            rpc.rpc_sync(to=send_cache_ref.owner(), func=_call_method, args=(KVCache.terminate,send_cache_ref),timeout=1)
        print("[CacheScheduler] 终止信号已发送")
