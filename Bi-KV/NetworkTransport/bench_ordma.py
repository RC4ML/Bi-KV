import argparse
import time
import torch
import torch.distributed as dist
import numpy as np
from rdma_onesided_transport import RDMAOneSidedEndpoint



def run_server(args, dist):
    # 构造服务器对象，mode 设置为 "server"
    server = RDMAOneSidedEndpoint(args.ip, args.port, "server")
    
    print("启动服务器，等待客户端连接...")
    # 这里假设仅接受一个客户端，local_cpu_size 和 local_gpu_size 单位均为字节
    ret = server.run_server(max_clients=1, 
                            local_cpu_size=args.server_cpu_size, 
                            local_gpu_size=args.server_gpu_size, 
                            hugepage=args.hugepage)
    if ret != 0:
        print("服务器启动失败")
        return

    print("客户端连接成功，开始测试 RDMA 操作性能...")
    server_gpu_tensor = server.get_server_gpu_tensor(rank=1)
    # 测试参数
    test_size = args.test_size  # 单次操作数据大小（字节）
    iterations = args.iterations
    # server_cpu_tensor.fill_(1)
    # print(server_cpu_tensor[:test_size+100]) #.fill_(1)  # 初始化服务器 CPU 内存 tensor
    dist.barrier()  # 确保所有进程同步
    # 对客户端（假设其 rank 为 1）执行 RDMA 写操作：将服务器本地内存写入客户端内存
    t0 = time.time()
    for i in range(iterations):
        ret = server.post_rdma_write(rank=1, size=test_size, src_type="gpu", dst_type="gpu")
        if ret != 0:
            print("post_rdma_write 失败")
            return
        # 轮询等待完成
        # 此处假设 post_rdma_write 内部会调用 poll_completion
    t1 = time.time()
    elapsed = t1 - t0
    throughput = (test_size * iterations) / (1024 * 1024) / elapsed
    print(f"RDMA write: {iterations} 次，总数据 {(test_size * iterations)/(1024*1024):.2f} MB, 用时 {elapsed:.4f} s, 吞吐量 {throughput:.2f} MB/s")

    # 同理测试 RDMA 读操作：从客户端内存读取到服务器本地内存
    t0 = time.time()
    for i in range(iterations):
        ret = server.post_rdma_read(rank=1, size=test_size, src_type="gpu", dst_type="gpu", local_offset=test_size)
        if ret != 0:
            print("post_rdma_read 失败")
            return
    t1 = time.time()
    elapsed = t1 - t0
    throughput = (test_size * iterations) / (1024 * 1024) / elapsed
    print(f"RDMA read: {iterations} 次，总数据 {(test_size * iterations)/(1024*1024):.2f} MB, 用时 {elapsed:.4f} s, 吞吐量 {throughput:.2f} MB/s")
    print(server_gpu_tensor[:test_size+100])
    # 如有需要，还可以获取对应的 Tensor
    gpu_tensor = server.get_server_gpu_tensor(rank=1)
    print("服务器 CPU tensor 大小：", gpu_tensor.size())

def run_client(args, dist):
    # 构造客户端对象，mode 设置为 "client"
    client = RDMAOneSidedEndpoint(args.ip, args.port, "client")
    print("客户端连接到服务器...")
    ret = client.connect_client(rank=args.rank, 
                                cpu_size=args.client_cpu_size, 
                                gpu_size=args.client_gpu_size, 
                                hugepage=args.hugepage)
    if ret != 0:
        print("客户端连接失败")
        return
    print("客户端连接成功，等待服务器发起 RDMA 操作...")
    tensor = client.get_client_gpu_tensor()
    tensor.fill_(3)  # 初始化客户端 CPU 内存 tensor
    print("客户端 CPU tensor 第一部分数据:", tensor[0:10].to('cpu').tolist())
    dist.barrier()  # 确保所有进程同步
    # 客户端主要等待服务器发起 RDMA 操作
    # 这里可以轮询打印客户端接收到的数据
    while True:
        try:
            # 获取客户端 CPU 内存 tensor
            # tensor = client.get_client_cpu_tensor()
            print("客户端 CPU tensor 第一部分数据:", tensor[0:10].to('cpu').tolist())
            time.sleep(2)
        except KeyboardInterrupt:
            break

def main():
    parser = argparse.ArgumentParser(description="测试单边 RDMA 性能")
    parser.add_argument("--role", type=str, choices=["server", "client"], required=True,
                        help="角色：server 或 client")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="服务器IP地址")
    parser.add_argument("--port", type=str, default="7471", help="端口号")
    parser.add_argument("--hugepage", action="store_true", help="是否使用大页内存")
    
    # 内存大小参数，单位字节
    parser.add_argument("--server_cpu_size", type=int, default=1024*1024*100,
                        help="服务器 CPU 内存大小（字节）")
    parser.add_argument("--server_gpu_size", type=int, default=0,
                        help="服务器 GPU 内存大小（字节），若不使用则设为0")
    parser.add_argument("--client_cpu_size", type=int, default=1024*1024*50,
                        help="客户端 CPU 内存大小（字节）")
    parser.add_argument("--client_gpu_size", type=int, default=0,
                        help="客户端 GPU 内存大小（字节），若不使用则设为0")
    
    # RDMA 操作性能测试参数
    parser.add_argument("--test_size", type=int, default=1024*1024,  # 1 MB
                        help="单次 RDMA 操作数据大小（字节）")
    parser.add_argument("--iterations", type=int, default=100,
                        help="测试迭代次数")
    
    # 客户端 rank id
    parser.add_argument("--rank", type=int, default=1, help="客户端 rank id")
    
    args = parser.parse_args()
    init_method = f"tcp://{args.ip}:{args.port}"
    if args.role == "server":
        rank = 0
    else:
        rank = args.rank
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=2,
        rank=rank
    )
    if args.role == "server":
        run_server(args, dist)
    elif args.role == "client":
        run_client(args, dist)

if __name__ == "__main__":
    main()
