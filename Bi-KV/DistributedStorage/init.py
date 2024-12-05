import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc
from cachescheduler import CacheScheduler
from kvcache import KVCache
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def init_backend(rank, world_size):
    """初始化 RPC"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 设置主节点的 IP 地址
    os.environ["MASTER_PORT"] = "29501"       # 设置主节点的通信端口
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"初始化GPU NCCL后端rank:{rank}")
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size
    )

    
def init_process(rank, world_size):
    """初始化每个进程"""
    init_backend(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    dist.barrier()
   
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
        ]
        scheduler.add_requests(requests)  # 一次性添加多个请求
        scheduler.process_requests()
        # # scheduler.send_terminate_signal()
        
    # dist.destroy_process_group()  # 清理分布式进程组, 注意要等rank0完成所有任务才能清理
    rpc.shutdown()  # 关闭 RPC

def main():
    """主函数"""
    print("[Main] 启动分布式系统")
    world_size = 5  # 动态获取可用 GPU 数量  4 GPU<kvcache> 1 CPU<cachescheduler>
    if world_size < 2:
        raise ValueError("需要至少 2 个 GPU 来运行此程序。")
    mp.spawn(init_process, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
