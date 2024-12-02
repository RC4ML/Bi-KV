# cachescheduler.py
import torch
import torch.distributed as dist
from threading import Lock, Thread
from Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
import time

class CacheScheduler:
    def __init__(self, world_size):
        """初始化调度器
        Args:
            world_size (int): 总进程数，包括调度器和所有 KVCache
        """
        print("[CacheScheduler] 初始化调度器")
        self.request_table = {}  # 用字典来存储请求表
        # 动态初始化 GPU 状态表，支持更多 GPU
        self.gpu_state_table = {rank: {'status': 'idle'} for rank in range(1, world_size)}
        self.lock = Lock()

    def add_requests(self, requests):
        """一次性添加多个请求到请求表
        Args:
            requests (list of tuples): 每个元组包含 (request_id, send_gpu, recv_gpu)
        """
        for request in requests:
            request_id, send_gpu, recv_gpu = request
            self.add_request(request_id, send_gpu, recv_gpu)

    def add_request(self, request_id, send_gpu, recv_gpu):
        """添加单个请求到请求表
        Args:
            request_id (int): 请求的唯一标识
            send_gpu (int): 发送数据的 GPU 编号
            recv_gpu (int): 接收数据的 GPU 编号
        """
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
                # 检查请求是否可执行
                if not executing and self.gpu_state_table.get(send_gpu, {}).get('status') == 'idle' and self.gpu_state_table.get(recv_gpu, {}).get('status') == 'idle':
                    with self.lock:
                        # 标记 GPU 状态为 busy
                        self.gpu_state_table[send_gpu]['status'] = 'sending'
                        self.gpu_state_table[recv_gpu]['status'] = 'receiving'
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
                send_gpu, recv_gpu = req['send_gpu'], req['recv_gpu']
                thread = Thread(target=self._execute_request, args=(request_id, send_gpu, recv_gpu))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

        print("[CacheScheduler] 所有请求处理完成")
        print(f"[CacheScheduler] 最终 GPU 状态表: {self.gpu_state_table}")
        return

    def _execute_request(self, request_id, send_gpu, recv_gpu):
        """执行单个请求
        Args:
            request_id (int): 请求的唯一标识
            send_gpu (int): 发送数据的 GPU 编号
            recv_gpu (int): 接收数据的 GPU 编号
        """
        print(f"[CacheScheduler] 执行请求 {request_id} - GPU {send_gpu} -> GPU {recv_gpu}")
        
        # 发送发送任务到发送GPU
        task_info_send = torch.tensor([SIGNAL_SEND, request_id, send_gpu, recv_gpu], device='cuda', dtype=torch.int32)
        dist.send(tensor=task_info_send, dst=send_gpu)

        # 发送接收任务到接收GPU
        task_info_recv = torch.tensor([SIGNAL_RECV, request_id, send_gpu, recv_gpu], device='cuda', dtype=torch.int32)
        dist.send(tensor=task_info_recv, dst=recv_gpu)

        # 等待接收GPU的确认消息
        confirmation_tensor = torch.zeros(1, device='cuda', dtype=torch.int32)
        dist.recv(tensor=confirmation_tensor, src=recv_gpu)

        if confirmation_tensor.item() == request_id:
            print(f"[CacheScheduler] 请求 {request_id} 完成 - GPU {send_gpu} -> GPU {recv_gpu}")

        with self.lock:
            del self.request_table[request_id]  # 删除请求
            self.gpu_state_table[send_gpu]['status'] = 'idle'
            self.gpu_state_table[recv_gpu]['status'] = 'idle'

    def send_terminate_signal(self):
        """发送终止信号给所有 KVCache"""
        print("[CacheScheduler] 发送终止信号给所有 KVCache")
        for gpu_rank in self.gpu_state_table.keys():
            task_info = torch.tensor([SIGNAL_TERMINATE, 0, 0, 0], device='cuda', dtype=torch.int32)
            dist.send(tensor=task_info, dst=gpu_rank)
        print("[CacheScheduler] 终止信号已发送")
