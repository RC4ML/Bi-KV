import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from cachescheduler import CacheScheduler
from kvcache import KVCache

def init_distributed_backend(rank, world_size):
    """初始化分布式环境"""
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"  # 设置主节点的 IP 地址
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"  # 设置主节点的通信端口
    # print(f"[Init] 初始化分布式环境，Rank={rank}, World Size={world_size}")
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dist.barrier()

def init_process(rank, world_size):
    """初始化每个进程"""
    init_distributed_backend(rank, world_size)
    
    if rank == 0:
        print(f"[Init][Rank {rank}] 初始化 CacheScheduler")
        scheduler = CacheScheduler()
        time.sleep(0.5)
        scheduler.add_request(1, 1, 2)
        scheduler.add_request(2, 2, 3)
        scheduler.add_request(3, 3, 1)
        scheduler.process_requests()
    else:
        print(f"[Init][Rank {rank}] 初始化 KVCache")
        kv_cache = KVCache(rank)
        kv_cache.run()
    

    
def main():
    """主函数"""
    print("[Main] 启动分布式系统")
    world_size = 4
    mp.spawn(init_process, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
