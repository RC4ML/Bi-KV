import grpc
import example_pb2
import example_pb2_grpc
import argparse
from rdma_transport import RDMAEndpoint
import time
import torch
import os
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

def run_rdma_client(server_ip, server_port, rank):
    print(f"Initializing RDMA Client (rank {rank})")
    client = RDMAEndpoint(server_ip, server_port, "client")
    
    if client.connect_client(rank) != 0:
        raise RuntimeError("RDMA connection failed")
    
    buffer_size = 1024*1024*128
    if client.register_memory_client(buffer_size) != 0:
        raise RuntimeError("Failed to register RDMA memory")
    

    
    return client

def run_grpc_client(rdma_client, grpc_ip, grpc_port, rank, buffer_size):
    # channel = grpc.insecure_channel(f'{grpc_ip}:{grpc_port}')
    
    with grpc.insecure_channel(f'{grpc_ip}:{grpc_port}') as channel:
        
        stub = example_pb2_grpc.RDMACommServiceStub(channel)
        start_time = time.time()

        response = stub.TriggerSend.future(example_pb2.TriggerRequest(
            rank=rank,
            buffer_size=buffer_size
        ))
        rdma_client.post_send(buffer_size)
        rdma_client.poll_completion()
        response.result()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        throughput = 1024*1024*128 / elapsed_time / (1024 * 1024)  # MB/s
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        print(f"Throughput: {throughput:.2f} MB/s")
        buffer_tensor = rdma_client.get_buffer_tensor()
        print(f"Received data: {buffer_tensor}")
        # if not response.success:
        #     raise RuntimeError(f"gRPC call failed: {response.message}")
        print("gRPC trigger successful")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdma_ip", required=True, help="RDMA Server IP")
    parser.add_argument("--rdma_port", required=True, help="RDMA Server Port")
    parser.add_argument("--grpc_ip", required=True, help="gRPC Server IP")
    parser.add_argument("--grpc_port", required=True, help="gRPC Server Port")
    parser.add_argument("--rank", type=int, default=0, help="Client rank")
    args = parser.parse_args()

    # 初始化 RDMA 客户端
    
    rdma_client = run_rdma_client(args.rdma_ip, args.rdma_port, args.rank)
    for _ in range(100):    
    # 发起 gRPC 调用
        print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.{int(time.time() * 1000) % 1000:03d}")
        run_grpc_client(rdma_client, args.grpc_ip, args.grpc_port, args.rank, 1024*1024*128)
    
