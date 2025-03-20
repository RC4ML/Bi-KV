import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import os


def init_process(rank, world_size):
    """初始化分布式进程和RPC"""
    # 初始化 NCCL 后端
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 设置主节点的 IP 地址
    os.environ["MASTER_PORT"] = "38656"       # 设置主节点的通信端口
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size
    )

def call_send_data(rref, dst):
    return rref.rpc_sync().send_data(dst)

def call_recv_data(rref, src):
    return rref.rpc_sync().recv_data(src)

def idle(rref, src):
    return rref.rpc_sync().idle()

class NCCLWorker:
    def __init__(self, rank):
        self.rank = rank

    def send_data(self, dst):
        """发送数据"""
        print(f"[Rank {self.rank}] Sending data to Rank {dst}")
        send_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=f"cuda:{self.rank}")

        dist.send(tensor=send_tensor, dst=dst)
        print(f"[Rank {self.rank}] Data sent to Rank {dst}")

    def recv_data(self, src):
        """接收数据"""
        print(f"[Rank {self.rank}] Receiving data from Rank {src}")
        received_tensor = torch.empty(4, device=f'cuda:{self.rank}')
        dist.recv(tensor=received_tensor, src=src)
        # received_tensor = received_tensor.to(f"cpu")
        print(received_tensor)
        print(f"[Rank {self.rank}] Data received from Rank {src}")

    def idle(self):
        print("haha")
        # while(1):
        #     sleep(1)

def main(rank, world_size):
    # 初始化分布式环境
    init_process(rank, world_size)
    dist.barrier() ## 重要
    if rank == 0:
        print("rank 0")
        worker1_ref = rpc.remote("worker1", NCCLWorker, args=(1,))
        worker2_ref = rpc.remote("worker2", NCCLWorker, args=(2,))
    
        future_recv = rpc.rpc_async(
            worker1_ref.owner(),
            call_recv_data,
            args=(worker1_ref, 2)
        )
        future_send = rpc.rpc_async(
            worker2_ref.owner(),
            call_send_data,
            args=(worker2_ref, 1)
        )
        
        future_send.wait()  # 确保任务完成
        future_recv.wait()  # 确保任务完成
        future_recv = rpc.rpc_async(
            worker1_ref.owner(),
            call_recv_data,
            args=(worker1_ref, 2)
        )
        future_send = rpc.rpc_async(
            worker2_ref.owner(),
            call_send_data,
            args=(worker2_ref, 1)
        )
        
        future_send.wait()  # 确保任务完成
        future_recv.wait()  # 确保任务完成
        future_recv = rpc.rpc_async(
            worker1_ref.owner(),
            call_recv_data,
            args=(worker1_ref, 2)
        )
        future_send = rpc.rpc_async(
            worker2_ref.owner(),
            call_send_data,
            args=(worker2_ref, 1)
        )
        
        future_send.wait()  # 确保任务完成
        future_recv.wait()  # 确保任务完成
    # else:
    #     print(f"rank{rank}")
    #     if rank == 1:
    #         remote_worker_ref = rpc.remote("worker2", NCCLWorker, args=(2,))
    #         future_recv = rpc.rpc_async(
    #             remote_worker_ref.owner(),
    #             call_recv_data,
    #             args=(remote_worker_ref, 1)
    #         )
    #         # dist.barrier() ## 重要
    #         future_recv.wait()
    #         # future_recv = rpc.rpc_async(
    #         #     remote_worker_ref.owner(),
    #         #     call_recv_data,
    #         #     args=(remote_worker_ref, 1)
    #         # )
    #         # future_recv.wait()
    #     else:
    #         remote_worker_ref = rpc.remote("worker1", NCCLWorker, args=(1,))
    #         future_send = rpc.rpc_async(
    #             remote_worker_ref.owner(),
    #             call_send_data,
    #             args=(remote_worker_ref, 2)
    #         )
    #         # dist.barrier() ## 重要
    #         future_send.wait()
    #         # future_send = rpc.rpc_async(
    #         #     remote_worker_ref.owner(),
    #         #     call_send_data,
    #         #     args=(remote_worker_ref, 2)
    #         # )
    #         # future_send.wait()
            
        # worker = NCCLWorker(rank)
        # if rank == 2:
        #     print(f"from rank{rank}")
        #     worker.send_data(1)
        #     worker.send_data(1)

        # else:
        #     print(f"to rank{rank}")
        #     worker.recv_data(2)
        #     worker.recv_data(2)
        # future_recv = rpc.rpc_async(
        #     worker1_ref.owner(),
        #     call_recv_data,
        #     args=(worker1_ref, 2)
        # )
        # future_send = rpc.rpc_async(
        #     worker2_ref.owner(),
        #     call_send_data,
        #     args=(worker2_ref, 1)
        # )
        
        # future_send.wait()  # 确保任务完成
        # future_recv.wait()  # 确保任务完成
    # dist.barrier() ## 重要

    # dist.destroy_process_group()
    # 清理环境
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 3
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)