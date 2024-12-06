import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc
from Scheduler import LLMScheduler
import warnings
from ComputeRun import run_scheduler
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 定义进程类型及其数量
PROCESS_TYPES = [
    ('scheduler', 1),
    ('coordinator', 1),
    ('inferworker', 4),
    ('kvcache', 4),
]

def get_process_info(rank, process_types=PROCESS_TYPES):
    """
    根据全局 rank 返回进程类型和该类型下的索引。

    Args:
        rank (int): 全局 rank。
        process_types (list): 进程类型及其数量的有序列表。

    Returns:
        tuple: (process_type, type_index)
    """
    current_rank = 0
    for process_type, count in process_types:
        if current_rank + count > rank:
            type_index = rank - current_rank
            return process_type, type_index
        current_rank += count
    raise ValueError(f"Rank {rank} 超出定义的进程类型范围。")

def init_backend(rank, world_size,process_type,type_index):
    """初始化 RPC 和分布式后端"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 设置主节点的 IP 地址
    os.environ["MASTER_PORT"] = "29501"       # 设置主节点的通信端口
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    # 初始化分布式进程组
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    print(f"[init_backend] 初始化 nccl backend rank {rank}")
    
    # 根据进程类型进行初始化
    if process_type == 'scheduler':
        print(f"[init_backend] scheduler{type_index}, CPU process")
        rpc.init_rpc(
            name=f"scheduler{type_index}",
            rank=rank,
            world_size=world_size
        )
    elif process_type == 'coordinator':
        print(f"[init_backend] coordinator{type_index}, CPU process")
        rpc.init_rpc(
            name=f"coordinator{type_index}",
            rank=rank,
            world_size=world_size
        )
    elif process_type == 'inferworker':
        torch.cuda.set_device(type_index)
        print(f"[init_backend] inferworker{type_index} 设置 GPU 索引: {type_index}")
        rpc.init_rpc(
            name=f"inferworker{type_index}",
            rank=rank,
            world_size=world_size
        )
    elif process_type == 'kvcache':
        torch.cuda.set_device(type_index)
        print(f"[init_backend] kvcache{type_index} 设置 GPU 索引: {type_index}")
        rpc.init_rpc(
            name=f"kvcache{type_index}",
            rank=rank,
            world_size=world_size
        )
    else:
        raise ValueError(f"未知的进程类型: {process_type}")

def init_process(rank, world_size):
    """初始化每个进程"""
    process_type, type_index = get_process_info(rank)
    init_backend(rank, world_size,process_type,type_index)
    
    if process_type.startswith('inferworker') or process_type.startswith('kvcache'):
        device = torch.device(f'cuda:{type_index}')
        torch.cuda.set_device(device)
    
    dist.barrier()
   
    if process_type == 'scheduler':
        print(f"[Init][Rank {rank}] 初始化 Scheduler{type_index}")        
        run_scheduler(world_size)
        
    
    # # 等待所有进程完成任务
    # dist.barrier()
    # # 销毁分布式进程组，注意要等rank0完成所有任务才能清理
    # dist.destroy_process_group()
    rpc.shutdown()  # 关闭 RPC

def main():
    """主函数"""
    print("[Main] 启动分布式系统")
    world_size = sum(count for _, count in PROCESS_TYPES)  # 根据定义的进程类型计算 world_size
    print(f"[Main] world_size = {world_size}，进程类型分布: {PROCESS_TYPES}")
    
    if world_size < 2:
        raise ValueError("需要至少 2 个进程来运行此程序。")
    
    mp.spawn(init_process, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
