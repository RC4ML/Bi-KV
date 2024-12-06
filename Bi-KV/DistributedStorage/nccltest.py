import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, world_size):
    """初始化分布式进程组"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    # 初始化分布式环境
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def run(rank, world_size):
    """每个进程的逻辑"""
    init_process(rank, world_size)

    # 每个进程绑定到对应的 GPU
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    if rank == 0:
        # Rank 0 发送数据
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        print(f"[Rank {rank}] Sending tensor: {tensor}")
        dist.send(tensor=tensor, dst=1)
    elif rank == 1:
        # Rank 1 接收数据
        tensor = torch.empty(4, device=device)
        print(f"[Rank {rank}] Waiting to receive tensor...")
        dist.recv(tensor=tensor, src=0)
        print(f"[Rank {rank}] Received tensor: {tensor}")
    
    # 清理分布式环境
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # 两个进程
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
