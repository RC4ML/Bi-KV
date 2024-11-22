import torch.distributed.rpc as rpc
import multiprocessing as mp

from Worker.Worker import Worker
from Scheduler.LLMScheduler import Scheduler

def timestamp_with_rref(user_rref):
    user = user_rref.to_here()
    item = user.timestamp
    id = user.user_id
    worker_id = int(rpc.get_worker_info().name[-1]) + 1  # 获取当前 worker 的 id
    return {"id":id,"ts":item,"worker_id":worker_id}

# 启动函数
def run_worker(rank, world_size):
    worker = Worker(rank,world_size)
    worker.start()
    worker.shutdown()

def run_scheduler(world_size):
    scheduler = Scheduler(num_workers=2,worker_func=timestamp_with_rref,world_size=world_size)
    scheduler.start()
    scheduler.shutdown()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    processes = []
    world_size = 4
    # 启动 workers
    for rank in range(1, world_size):  # 假设有 2 个 worker
        p = mp.Process(target=run_worker, args=(rank, world_size))
        p.start()
        processes.append(p)

    # 启动 scheduler
    p = mp.Process(target=run_scheduler,args=(world_size,))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()






