import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc
import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc
from Scheduler.LLMScheduler import LLMScheduler
from inputGenerator.inputGenerator import LLMInput

from config import *
args.model_code = 'llm'
# args.llm_retrieved_path = "/share/gnn_data/testmodel/LlamaRec/experiments/lru/games/"
args.llm_retrieved_path = "/data/testmodel/LlamaRec/experiments/lru/games"
args.dataset_code = "games"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def init_rpc_backend(rpc_name, rank, world_size):
    """初始化 RPC"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 设置主节点的 IP 地址
    os.environ["MASTER_PORT"] = "29500"       # 设置主节点的通信端口
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    rpc.init_rpc(name=rpc_name, rank=rank, world_size=world_size)
    print(f"[init_rpc_backend]rpc初始化{rpc_name}")

def run_worker(rank, world_size):
    """初始化每个进程"""
    if rank == 0:
        init_rpc_backend(f"LLMScheduler{rank}", rank, world_size)
        llm_input = LLMInput(20,500,args)
        llm_input.set_random("weighted")
        generate_res = llm_input.Generate(100)
        scheduler = LLMScheduler(world_size=world_size)
        time.sleep(1)
        scheduler.add_prompt_list(generate_res)
        scheduler.process_prompt()
        #scheduler.send_terminate_signal()
    else:
        init_rpc_backend(f"Worker{rank}", rank, world_size)

    # 确保在销毁进程组之前，RPC 完成了所有操作
    rpc.shutdown()  # 关闭 RPC
    if rank == 0:
        print(f"LLMScheduler结束rank={rank}")
    else:
        print(f"Worker结束rank={rank}")

def main():
    """主函数"""
    print("[Main] 启动分布式系统")
    world_size = 5  # 动态获取可用 GPU 数量  4 GPU<kvcache> 1 CPU<CacheCoordinator>
    if world_size < 2:
        raise ValueError("需要至少 2 个 GPU 来运行此程序。")
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
