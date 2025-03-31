import os
import time
import argparse
import torch
import torch.distributed as dist

def setup(rank, world_size, master_addr, master_port, backend='gloo'):
    """
    初始化分布式环境。
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    """
    清理分布式环境。
    """
    dist.destroy_process_group()

def run_send_recv(rank, world_size, tensor_size, backend='gloo', num_iter=10):
    """
    实现 send 和 recv 操作，并测试性能。
    """
    # 创建张量
    if backend == 'nccl':
        device = torch.device(f'cuda')  # 使用 GPU
    else:
        device = torch.device('cpu')  # 使用 CPU

    if rank == 0:
        tensor = torch.ones(tensor_size, device=device)  # 发送方
    else:
        tensor = torch.zeros(tensor_size, device=device)  # 接收方

    # 预热（避免第一次运行时的额外开销）
    for _ in range(2):
        if rank == 0:
            dist.send(tensor=tensor, dst=1)
        elif rank == 1:
            dist.recv(tensor=tensor, src=0)
        dist.barrier()

    # 测试性能
    total_time = 0.0
    for _ in range(num_iter):
        dist.barrier()  # 同步所有进程
        start_time = time.time()

        if rank == 0:
            dist.send(tensor=tensor, dst=1)  # 发送数据
        elif rank == 1:
            dist.recv(tensor=tensor, src=0)  # 接收数据

        dist.barrier()  # 同步所有进程
        end_time = time.time()
        total_time += (end_time - start_time)

    # 计算吞吐量
    if rank == 0:
        data_size_bytes = tensor.nelement() * tensor.element_size()  # 数据大小（字节）
        total_data_size_gb = data_size_bytes * num_iter / (1024 ** 3)  # 总数据大小（GB）
        throughput_gb_per_sec = total_data_size_gb / total_time  # 吞吐量（GB/s）
        print(f"Throughput: {throughput_gb_per_sec:.4f} GB/s")
        print(f"Average communication time per iteration: {total_time / num_iter:.4f} seconds")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="PyTorch Distributed Send/Recv Performance Test")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current process")
    parser.add_argument("--world_size", type=int, required=True, help="Total number of processes")
    parser.add_argument("--master_addr", type=str, default="192.168.189.8", help="Master node IP address")
    parser.add_argument("--master_port", type=str, default="29500", help="Master node port")
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"], help="Backend to use (gloo or nccl)")
    parser.add_argument("--tensor_size", type=int, default=8192, help="Size of the tensor (e.g., 1024 for a 1MB tensor)")
    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations for performance testing")
    args = parser.parse_args()

    # 初始化分布式环境
    setup(args.rank, args.world_size, args.master_addr, args.master_port, args.backend)

    # 运行 send 和 recv 测试
    tensor_size = (args.tensor_size, args.tensor_size)  # 创建指定大小的张量
    run_send_recv(args.rank, args.world_size, tensor_size, args.backend, args.num_iter)

    # 清理分布式环境
    cleanup()