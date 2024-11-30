# main.py
import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from cachescheduler import CacheScheduler
from kvcache import KVCache

def init_distributed_backend(rank, world_size):
    """初始化分布式环境"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 设置主节点的 IP 地址
    os.environ["MASTER_PORT"] = "29500"       # 设置主节点的通信端口
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dist.barrier()

def init_process(rank, world_size):
    """初始化每个进程"""
    init_distributed_backend(rank, world_size)
    
    if rank == 0:
        print(f"[Init][Rank {rank}] 初始化 CacheScheduler")
        scheduler = CacheScheduler(world_size)
        time.sleep(0.5)
        
        # 定义多个请求
        requests = [
            (1, 1, 2),
            (2, 3, 4),
            (3, 3, 1),
            (4, 1, 3),
            (5, 2, 1)
            # 可以在这里添加更多请求
        ]
        scheduler.add_requests(requests)  # 一次性添加多个请求
        scheduler.process_requests()
        scheduler.send_terminate_signal()
    else:
        print(f"[Init][Rank {rank}] 初始化 KVCache")
        kv_cache = KVCache(rank)
        kv_cache.run()
    dist.barrier()
    dist.destroy_process_group()  # 清理分布式进程组

def main():
    """主函数"""
    print("[Main] 启动分布式系统")
    world_size = 5  # 动态获取可用 GPU 数量
    if world_size < 2:
        raise ValueError("需要至少 2 个 GPU 来运行此程序。")
    mp.spawn(init_process, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
