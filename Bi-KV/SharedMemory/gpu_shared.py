import numpy as np
import torch
import pycuda.driver as cuda
from pycuda import autoinit
import multiprocessing as mp
import time

DEBUG = False

def _get_gpu_memory(gpu_index):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_index}')
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        return f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB"
    return "N/A"

class Writer:
    def __init__(self, gpu_index, rank, mb_size):
        self.gpu_index = gpu_index
        self.ctx = cuda.Device(gpu_index).make_context()
        self.rank = rank
        self.device = torch.device(f'cuda:{gpu_index}')
        self.mb_size = mb_size

    @staticmethod
    def allocate_gpu_tensor(mb_size, device, dtype=torch.float16):
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        total_bytes = mb_size * 1024**2
        num_elements = total_bytes // bytes_per_element
        return torch.empty(num_elements, dtype=dtype, device=device).contiguous()

    def share_gpu_memory(self):
        if DEBUG:
            print(f"Writer 创建张量前显存: {_get_gpu_memory(self.gpu_index)}")
        send_tensor = self.allocate_gpu_tensor(self.mb_size, self.device)
        send_tensor.fill_(1.0)
        if DEBUG:
            print(f"Writer 创建张量后显存: {_get_gpu_memory(self.gpu_index)}")
        try:
            write_start_time = time.time()
            self.ctx.push()
            ptr = send_tensor.data_ptr()
            handle = cuda.mem_get_ipc_handle(ptr)
            write_end_time = time.time()
            write_time = write_end_time - write_start_time
            # with open(f'/data/zzm/Bi-KV/test_result/shared_memory_test/gpu_shared_test{self.mb_size}MB.txt', mode='a+') as f:
            #     f.write(f"[Create] 共享内存创建时间: {write_time:.4f}s\n")
            return write_start_time, write_time, handle, send_tensor.shape
        finally:
            self.ctx.pop()

class Reader:
    def __init__(self, gpu_index, rank, mb_size):
        self.gpu_index = gpu_index
        self.ctx = cuda.Device(gpu_index).make_context()
        self.rank = rank
        self.device = torch.device(f'cuda:{gpu_index}')
        self.mb_size = mb_size

    def _read_from_shared_memory_batch_gpu(self, remote_handle, shape):
        read_start_time=time.time()
        self.ctx.push()
        try:
            ipc_handle = cuda.IPCMemoryHandle(remote_handle)
            ptr = int(ipc_handle)
            gpu_array = cuda.from_device(ptr, shape, np.float16, order='C')
            recv_tensor = torch.as_tensor(gpu_array, device=self.device)
            read_end_time=time.time()
            copy_tensor = recv_tensor.clone()
            copy_end_time = time.time()
            return copy_tensor, copy_end_time - read_end_time,read_end_time-read_start_time
        finally:
            self.ctx.pop()
            ipc_handle.close()

def writer_process(queue, gpu_index, mb_size):
    writer = Writer(gpu_index, 0, mb_size)
    start_time, create_time, handle_bytes, shape = writer.share_gpu_memory()
    queue.put((handle_bytes, shape, start_time, create_time))
    time.sleep(5)
    writer.ctx.pop()

def reader_process(queue, gpu_index, mb_size, result_queue):
    reader = Reader(gpu_index, 1, mb_size)
    handle_bytes, shape, start_time, create_time = queue.get()
    
    # 记录传输开始时间
    transfer_start = time.time()
    
    # 执行实际读取操作
    recv_tensor, copy_time,read_time = reader._read_from_shared_memory_batch_gpu(handle_bytes, shape)
    
    # 计算各时间指标
    end_to_end_duration = time.time() - start_time  # 总端到端时间
    communicate_time = end_to_end_duration - read_time - create_time-copy_time  # 通信开销
    
    # 数据验证
    expected = torch.ones(shape, dtype=torch.float16, device=reader.device)
    assert torch.allclose(recv_tensor, expected, atol=1e-5), "数据验证失败!"
    
    # 计算所有速率指标
    def safe_div(a, b):
        return a / b if b > 1e-6 else 0
    
    rate_read = safe_div(mb_size, read_time)              # 读取阶段速率
    rate_e2e = safe_div(mb_size, end_to_end_duration)     # 端到端速率
    rate_comm = safe_div(mb_size, communicate_time)       # 通信阶段速率
    rate_create = safe_div(mb_size, create_time)          # 创建阶段速率
    rate_copy = safe_div(mb_size, copy_time)
    
    # 输出并记录详细指标
    log_content = (
        f"[Result] 创建时间: {create_time:.4f}s | "
        f"端到端: {end_to_end_duration:.4f}s | "
        f"读取时间: {read_time:.4f}s | "
        f"通信时间: {communicate_time:.4f}s\n"
        f"copy时间: {copy_time:.4f}s\n"
        f"[Rate] 创建速率: {rate_create:.2f}MB/s | "
        f"端到端速率: {rate_e2e:.2f}MB/s | "
        f"读取速率: {rate_read:.2f}MB/s | "
        f"通信速率: {rate_comm:.2f}MB/s\n"
        f"copy速率: {rate_copy:.2f}MB/s\n"
    )
    print(log_content)
    
    with open(f'/data/zzm/Bi-KV/test_result/shared_memory_test/gpu_shared_test{mb_size}MB.txt', mode='a+') as f:
        f.write(log_content)
    
    # 将所有指标打包发送
    result_queue.put((
        create_time, end_to_end_duration, 
        read_time, communicate_time,copy_time,
        rate_create, rate_e2e, rate_read, rate_comm,rate_copy
    ))

if __name__ == "__main__":
    mp.set_start_method('spawn')
    queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 配置测试参数
    GPU_INDEX = 0
    MB_SIZE = 800   # 测试数据大小
    NUM_TESTS = 10    # 测试次数
    
    # 初始化结果统计
    metrics = {
        'create_time': 0.0,
        'end_to_end': 0.0,
        'read_time': 0.0,
        'comm_time': 0.0,
        'copy_time': 0.0,
        'rate_create': 0.0,
        'rate_e2e': 0.0,
        'rate_read': 0.0,
        'rate_comm': 0.0,
        'rate_copy': 0.0
    }
    
    # 执行多轮测试
    for i in range(NUM_TESTS):
        writer = mp.Process(target=writer_process, args=(queue, GPU_INDEX, MB_SIZE))
        reader = mp.Process(target=reader_process, args=(queue, GPU_INDEX, MB_SIZE, result_queue))
        
        writer.start()
        reader.start()
        
        writer.join()
        reader.join()
        
        if i>4:
            # 收集单次测试结果
            if not result_queue.empty():
                data = result_queue.get()
                metrics['create_time'] += data[0]
                metrics['end_to_end'] += data[1]
                metrics['read_time'] += data[2]
                metrics['comm_time'] += data[3]
                metrics['copy_time'] += data[4]
                metrics['rate_create'] += data[5]
                metrics['rate_e2e'] += data[6]
                metrics['rate_read'] += data[7]
                metrics['rate_comm'] += data[8]
                metrics['rate_copy'] += data[9]
    
    # 计算平均值
    avg_metrics = {k: v/5 for k, v in metrics.items()}
    
    # 生成最终报告
    time_report = (
        f"\n[平均时间] 创建: {avg_metrics['create_time']:.4f}s | "
        f"端到端: {avg_metrics['end_to_end']:.4f}s\n"
        f"读取: {avg_metrics['read_time']:.4f}s | "
        f"通信: {avg_metrics['comm_time']:.4f}s\n"
        f"copy: {avg_metrics['copy_time']:.4f}s\n"
    )
    rate_report = (
        f"[平均速率] 创建阶段: {avg_metrics['rate_create']:.2f}MB/s\n"
        f"端到端速率: {avg_metrics['rate_e2e']:.2f}MB/s | "
        f"读取速率: {avg_metrics['rate_read']:.2f}MB/s\n"
        f"通信速率: {avg_metrics['rate_comm']:.2f}MB/s\n"
        f"copy速率: {avg_metrics['rate_copy']:.2f}MB/s\n"
    )
    
    print(time_report)
    print(rate_report)
    
    with open(f'/data/zzm/Bi-KV/test_result/shared_memory_test/gpu_shared_test{MB_SIZE}MB.txt', mode='a+') as f:
        f.write(time_report)
        f.write(rate_report)