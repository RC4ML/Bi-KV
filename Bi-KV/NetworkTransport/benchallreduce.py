import torch
import torch.distributed as dist
import argparse
import os 

def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch NCCL all_reduce test across multiple machines."
    )
    parser.add_argument("--master_addr", type=str, default="10.0.0.3",
                        help="IP address of the master node.")
    parser.add_argument("--master_port", type=str, default="12345",
                        help="Port number used by the master node.")
    parser.add_argument("--world_size", type=int, default=4,
                        help="Total number of processes across all nodes.")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank of the current process.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank (if using multiple GPUs on one node).")
    return parser.parse_args()

def main():
    args = parse_args()
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ.get('RANK', os.environ.get('OMPI_COMM_WORLD_RANK', -1)))
    # 初始化进程组
    print(f"Rank {rank} start")
    init_method = f"tcp://{args.master_addr}:{args.master_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )

    # 为了使用 NCCL，需要使用 GPU 张量并指定正确的设备
    # 如果每台机器只有1块GPU，可直接默认使用 local_rank=0
    # 多GPU时，可以使用 --local_rank=xxx, 这样每个进程绑定不同 GPU
    torch.cuda.set_device(args.local_rank)

    # 每个进程创建一个张量，内容为 [rank+1]，放到指定 GPU 上
    tensor = torch.tensor([rank + 1], dtype=torch.float32).cuda()

    print(f"[Before all_reduce] Rank {rank} has data: {tensor.item()}")
    dist.barrier()
    # 执行 all_reduce 操作，使用 SUM
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"[After all_reduce] Rank {rank} has data: {tensor.item()}")

    # 销毁进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
