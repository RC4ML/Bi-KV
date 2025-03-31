import grpc
from concurrent import futures
import example_pb2
import example_pb2_grpc
import argparse
from rdma_transport import RDMAEndpoint
import time
import threading
import torch
import os
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

grpc_ready = threading.Event()
class RDMACommServicer(example_pb2_grpc.RDMACommServiceServicer):
    def __init__(self, rdma_server):
        self.rdma_server = rdma_server
        
    def TriggerSend(self, request, context):
        rank = request.rank
        buffer_size = request.buffer_size
        
        for _ in range(1):
            self.rdma_server.post_receive_by_rank(rank)
            self.rdma_server.poll_completion_by_rank(rank)
        try:
            # 发送 RDMA 数据
            print(f"Triggering RDMA send to rank {rank}")
            return example_pb2.TriggerResponse(
                success=True,
                message=f"Data sent to rank {rank}"
            )
        except Exception as e:
            return example_pb2.TriggerResponse(
                success=False,
                message=str(e)
            )

def run_grpc_server(ip, port, rdma_server):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    example_pb2_grpc.add_RDMACommServiceServicer_to_server(
        RDMACommServicer(rdma_server), server)
    server.add_insecure_port(f'{ip}:{port}')
    server.start()
    print(f"gRPC Server started on {ip}:{port}")
    grpc_ready.set()  # 标记 gRPC 服务器已启动
    server.wait_for_termination()

def run_rdma_server(rdma_server, ip, port, max_clients):
    print(f"Initializing RDMA Server on {ip}:{port}")
    
    if rdma_server.run_server(max_clients) != 0:
        raise RuntimeError("RDMA Server failed to start")
    
    buffer_size = 1024*1024*128
    for rank in range(max_clients):
        if rdma_server.register_memory(rank, buffer_size) != 0:
            raise RuntimeError(f"Failed to register memory for rank {rank}")
    
    # 初始化数据
    for rank in range(max_clients):
        buffer_tensor = rdma_server.get_buffer_tensor_by_rank(rank)
        buffer_tensor.fill_(rank+1)
    
    return rdma_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdma_ip", required=True, help="RDMA Server IP")
    parser.add_argument("--rdma_port", required=True, help="RDMA Server Port")
    parser.add_argument("--grpc_ip", required=True, help="gRPC Server IP")
    parser.add_argument("--grpc_port", required=True, help="gRPC Server Port")
    parser.add_argument("--clients", type=int, default=1, help="Number of clients")
    args = parser.parse_args()

    # 先初始化 RDMA
    
    # 在独立线程中运行 gRPC 服务
    rdma_server = RDMAEndpoint(args.rdma_ip, args.rdma_port, "server")


    grpc_thread = threading.Thread(
        target=run_grpc_server,
        args=(args.grpc_ip, args.grpc_port, rdma_server)
    )
    grpc_thread.start()
    grpc_ready.wait()
    rdma_server = run_rdma_server(rdma_server, args.rdma_ip, args.rdma_port, args.clients)
    
    buffer_tensor = rdma_server.get_buffer_tensor_by_rank(0)
    buffer_tensor.fill_(2)
    
    
    time.sleep(50)
    # for _ in range(100):
    #     # print("send once")
    #     rdma_server.post_send_by_rank(0, 1024*1024*128)
    #     rdma_server.poll_completion_by_rank(0)
    # rdma_thread = threading.Thread(
    #     target=run_rdma_server,
    #     args=(args.rdma_ip, args.rdma_port, args.clients)
    # )
    # rdma_thread.start()

    # try:
    #     while True:
    #         time.sleep(3600)
    # except KeyboardInterrupt:
    #     print("Shutting down servers...")