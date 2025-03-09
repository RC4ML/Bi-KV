import numpy as np
import torch
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import time
import uuid

DEBUG = True

class Writer:
    def __init__(self, rank, mb_size):
        self.rank = rank
        self.mb_size = mb_size

    @staticmethod
    def allocate_cpu_tensor(mb_size, dtype=torch.float16):
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        total_bytes = mb_size * 1024**2
        num_elements = total_bytes // bytes_per_element
        return torch.empty(num_elements, dtype=dtype).contiguous()

    def share_cpu_memory(self):
        if DEBUG:
            print(f"Writer 创建张量前内存状态...")
        send_tensor = self.allocate_cpu_tensor(self.mb_size)
        send_tensor.fill_(1.0)
        if DEBUG:
            print(f"Writer 张量大小: {send_tensor.nbytes / 1024**2:.2f} MB")

        # 创建唯一名称的共享内存
        shm_name = f'sm_{uuid.uuid4().hex}'
        start_time=time.time()
        shm = SharedMemory(create=True, size=send_tensor.nbytes, name=shm_name)
        # 将数据拷贝到共享内存
        torch_buffer = torch.frombuffer(shm.buf, dtype=send_tensor.dtype).reshape(send_tensor.shape)
        torch_buffer.copy_(send_tensor)
        return start_time, shm_name, send_tensor.shape, shm

class Reader:
    def __init__(self, rank, mb_size):
        self.rank = rank
        self.mb_size = mb_size

    def _read_from_shared_memory(self, shm_name, shape):
        shm = SharedMemory(name=shm_name)
        try:
            recv_tensor = torch.frombuffer(shm.buf, dtype=torch.float16).reshape(shape).clone()
            return recv_tensor
        finally:
            shm.close()

def writer_process(queue, mb_size):
    writer = Writer(0, mb_size)
    start_time, shm_name, shape, shm = writer.share_cpu_memory()
    queue.put((shm_name, shape, start_time))
    # 等待足够时间确保reader读取完成
    time.sleep(10)
    # 关闭并释放共享内存
    shm.close()
    shm.unlink()
    if DEBUG:
        print(f"Writer 释放共享内存: {shm_name}")

def reader_process(queue, mb_size, result_queue):
    reader = Reader(1, mb_size)
    shm_name, shape, start_time = queue.get()
    transfer_start = time.time()
    try:
        recv_tensor = reader._read_from_shared_memory(shm_name, shape)
    except FileNotFoundError:
        result_queue.put(0)
        return
    
    transfer_duration = time.time() - start_time
    # 验证数据正确性
    expected = torch.ones(shape, dtype=torch.float16)
    assert torch.allclose(recv_tensor, expected, atol=1e-5), "数据校验失败"
    # 计算传输速率
    operational_duration = time.time() - transfer_start
    rate = mb_size / operational_duration
    print(f"传输成功! 速率: {rate:.2f} MB/s | 端到端延迟: {transfer_duration:.4f}s")
    with open(f'/data/zzm/Bi-KV/test_result/shared_memory_test/cpu_shared_test{mb_size}MB.txt', mode='a+') as f:
        f.write(f"传输成功! 速率: {rate:.2f} MB/s | 端到端延迟: {transfer_duration:.4f}s\n")
    result_queue.put(rate)

if __name__ == "__main__":
    
    mp.set_start_method('spawn')
    queue = mp.Queue()
    result_queue = mp.Queue()
    
    MB_SIZE = 3000
    NUM_TESTS = 5
    with open(f'/data/zzm/Bi-KV/test_result/shared_memory_test/cpu_shared_test{MB_SIZE}MB.txt', 'a+') as f:
        f.write(f"CPU shared_memory {MB_SIZE}MB test\n")
    
    total_rate = 0
    for _ in range(NUM_TESTS):
        writer = mp.Process(target=writer_process, args=(queue, MB_SIZE))
        reader = mp.Process(target=reader_process, args=(queue, MB_SIZE, result_queue))
        
        writer.start()
        reader.start()
        
        writer.join()
        reader.join()
        
        if not result_queue.empty():
            total_rate += result_queue.get()
    
    avg_rate = total_rate / NUM_TESTS
    print(f"平均传输速率: {avg_rate:.2f} MB/s")
    with open(f'/data/zzm/Bi-KV/test_result/shared_memory_test/cpu_shared_test{MB_SIZE}MB.txt', 'a+') as f:
        f.write(f"平均速率: {avg_rate:.2f} MB/s\n")