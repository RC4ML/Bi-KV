import torch.distributed.rpc as rpc

# Worker 类
class Worker:
    def __init__(self, rank,world_size):
        self.rank = rank
        self.name = f"worker{rank - 1}"
        self.world_size = world_size
    
    def start(self):
        options = rpc.TensorPipeRpcBackendOptions(init_method='tcp://localhost:29500', num_worker_threads=256)
        rpc.init_rpc(
            name=self.name,
            rank=self.rank,
            world_size=self.world_size,  # 假设有 2 个 worker 和 1 个 master
            rpc_backend_options=options
        )
        print(f"{self.name} initialized")
        pass
    
    def shutdown(self):
        rpc.shutdown()