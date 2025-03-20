import argparse
import time
import torch
from rdma_transport import RDMAEndpoint

def run_client(ip, port, rank):
    print(f"Client {rank} connecting to {ip}:{port}...")

    # 创建 RDMA 客户端
    client = RDMAEndpoint(ip, port, "client")

    # 连接到服务器并发送 `rank`
    if client.connect_client(rank) != 0:
        print("Client connection failed!")
        return
    
    print(f"Client {rank} connected!")

    # 注册 RDMA 内存
    buffer_size = 1024*1024*128
    if client.register_memory_client(buffer_size) != 0:
        print("Failed to register memory!")
        return
    
    start_time = time.time()

    for _ in range(100):
        # 发布接收请求
        client.post_receive()
        
        # 等待服务器发送数据
        client.poll_completion()

    end_time = time.time()
    elapsed_time = end_time - start_time
    throughput = buffer_size * 100 / elapsed_time / (1024 * 1024)  # MB/s

    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f"Throughput: {throughput:.2f} MB/s")
    
    # 读取接收的 RDMA 缓冲区数据
    buffer_tensor = client.get_buffer_tensor()
    print(f"Client {rank} received data: {buffer_tensor}")

    time.sleep(2)
    print(f"Client {rank} shutting down.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDMA Client")
    parser.add_argument("--ip", type=str, required=True, help="Server IP")
    parser.add_argument("--port", type=str, required=True, help="Server Port")
    parser.add_argument("--rank", type=int, required=True, help="Client rank ID")
    args = parser.parse_args()

    run_client(args.ip, args.port, args.rank)
