import argparse
import time
from rdma_transport import RDMAEndpoint
import torch

def run_server(ip, port, max_clients):
    print(f"Starting RDMA Server on {ip}:{port}, waiting for {max_clients} clients...")
    
    # 创建 RDMA 服务器
    server = RDMAEndpoint(ip, port, "server")

    # 运行服务器，接受 `max_clients` 个连接
    if server.run_server(max_clients) != 0:
        print("Server failed to accept clients.")
        return
    
    print("All clients connected successfully!")

    # 为所有客户端分配 RDMA 缓冲区
    buffer_size = 1024*1024*128  # 每个客户端分配 1KB
    for rank in range(max_clients):
        if server.register_memory(rank, buffer_size) != 0:
            print(f"Failed to register memory for client {rank}")
            return
    for rank in range(max_clients):
        buffer_tensor = server.get_buffer_tensor_by_rank(rank)
        buffer_tensor.fill_(rank+1)
        print(f"client {rank} will receive data: {buffer_tensor}")
    
    start_time = time.time()
    for _ in range(100):
        # 向所有客户端发送数据
        for rank in range(max_clients):
            print(f"Server posting send request to rank {rank}...")
            server.post_send_by_rank(rank, buffer_size)
        
        # 等待所有传输完成
        for rank in range(max_clients):
            server.poll_completion_by_rank(rank)
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = (buffer_size * 100 * max_clients) / total_time / (1024 * 1024)  # MB/s
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} MB/s")
    
    print("Data sent to all clients!")
    time.sleep(2)  # 确保客户端有足够时间接收数据
    print("Server shutting down.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDMA Server")
    parser.add_argument("--ip", type=str, required=True, help="Server IP")
    parser.add_argument("--port", type=str, required=True, help="Server Port")
    parser.add_argument("--clients", type=int, required=True, help="Max number of clients")
    args = parser.parse_args()

    run_server(args.ip, args.port, args.clients)
