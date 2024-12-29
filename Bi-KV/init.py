import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用所有GPU
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc
from Scheduler.LLMScheduler import LLMScheduler
from DistributedStorage.CacheCoordinator import CacheCoordinator
from inputGenerator.inputGenerator import LLMInput
import warnings
import logging
from config import *
from rpc_def import PROCESS_TYPES, KVCACHE_NUM, WORKER_NUM, get_process_info
from Remote.remote_call import call_remote_method
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

args.model_code = 'llm'
args.llm_retrieved_path =  "/share/nfs/sunjie/games"
args.dataset_code = "games"

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("distributed_system.log"),
        logging.StreamHandler()
    ]
)

def init_backend(rank, world_size, process_type, type_index, timeout = 120):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    
    logging.info(f"[init_backend] 初始化进程类型: {process_type}{type_index}, Rank: {rank}")
    
    rpc.init_rpc(
        name=f"{process_type}{type_index}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            rpc_timeout = timeout#, _transports=["uv"]
        )
    )

def init_process(rank, world_size):
    process_type, type_index = get_process_info(rank, PROCESS_TYPES)
    if process_type == 'scheduler':
        timeout = 180
    else:
        timeout = 60
    init_backend(rank, world_size, process_type, type_index, timeout=timeout)
    dist.barrier()

    if process_type == 'scheduler':
        logging.info(f"[init_process][Rank {rank}] 初始化 LLMScheduler")
        scheduler = LLMScheduler(world_size=world_size)
        # scheduler.test_write_cache()
        input_generator = LLMInput(20,5,args)
        # input_generator.set_random('random')
        logging.info("开始测试")
        scheduler.set_prompt_generator(input_generator)
        scheduler.start(4,128)

    dist.barrier()
    rpc.shutdown()
    dist.destroy_process_group()

def main():
    logging.info("[Main] 启动分布式系统")
    world_size = sum(count for _, count in PROCESS_TYPES)
    logging.info(f"[Main] world_size = {world_size}，进程类型分布: {PROCESS_TYPES}")
    
    if world_size < 2:
        raise ValueError("需要至少 2 个进程来运行此程序。")
    
    mp.spawn(init_process, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
