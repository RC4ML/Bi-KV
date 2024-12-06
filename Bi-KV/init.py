"init.py"
import os
# 禁用GPU (确保程序全局使用CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用所有GPU
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc
from Scheduler.LLMScheduler import LLMScheduler
from inputGenerator.inputGenerator import LLMInput
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from config import *
args.model_code = 'llm'
# args.llm_retrieved_path = "/share/gnn_data/testmodel/LlamaRec/experiments/lru/games/"
args.llm_retrieved_path = "/data/testmodel/LlamaRec/experiments/lru/games"
args.dataset_code = "games"
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from rpc_def import *

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("distributed_system.log"),
        logging.StreamHandler()
    ]
)


# 定义进程类型及其数量
PROCESS_TYPES = [
    ('scheduler', 1),
    ('coordinator', 1),
    ('inferworker', 4),
    ('kvcache', 4),
]

def _call_cordinator_process(rref):
    return rref.rpc_sync().process_requests()
def _call_terminate_process(rref):
    return rref.rpc_sync().send_terminate_signal()
def get_process_info(rank, process_types=PROCESS_TYPES):
    """
    根据全局 rank 返回进程类型和该类型下的索引。

    Args:
        rank (int): 全局 rank。
        process_types (list): 进程类型及其数量的有序列表。

    Returns:
        tuple: (process_type, type_index)
    """
    current_rank = 0
    for process_type, count in process_types:
        if current_rank + count > rank:
            type_index = rank - current_rank
            return process_type, type_index
        current_rank += count
    raise ValueError(f"Rank {rank} 超出定义的进程类型范围。")

def init_backend(rank, world_size, process_type, type_index):
    """初始化 RPC 和分布式后端"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 设置主节点的 IP 地址
    os.environ["MASTER_PORT"] = "29501"       # 设置主节点的通信端口
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    # 初始化分布式进程组
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    
    # 根据进程类型进行初始化
    logging.info(f"[init_backend] 初始化进程类型: {process_type}{type_index}, Rank: {rank}")
    
    rpc.init_rpc(
        name=f"{process_type}{type_index}",
        rank=rank,
        world_size=world_size
    )

def init_process(rank, world_size):
    """初始化每个进程"""
    process_type, type_index = get_process_info(rank)
    init_backend(rank, world_size, process_type, type_index)
    dist.barrier()

    if process_type == 'scheduler':
        logging.info(f"[init_process][Rank {rank}] 初始化 LLMScheduler")
        scheduler = LLMScheduler(world_size=world_size)

        generate_res = [        
            (1, 1, 2),
            # (2, 3, 0),
            (3, 3, 1),
            (4, 1, 3),
            (5, 2, 1)
        ]
        scheduler.add_prompt_list(generate_res)
        logging.info("开始测试")
        scheduler.process_prompt()

        # 输出检查：等待协调器处理请求并检查结果
        future_call_coordin_process = rpc.rpc_sync(
            scheduler.coordinator_ref[0].owner(),
            _call_cordinator_process,
            args=(scheduler.coordinator_ref[0],)
        )
        future_call_terminate_process = rpc.rpc_sync(
            scheduler.coordinator_ref[0].owner(),
            _call_terminate_process,
            args=(scheduler.coordinator_ref[0],)
        )

        # 这里可以添加更多的输出检查逻辑，例如验证结果是否符合预期
        # if result != expected_result:
        #     logging.error("输出检查失败！")
        # else:
        #     logging.info("输出检查通过。")

    # 等待所有进程完成任务
    # dist.barrier()
    # 销毁分布式进程组
    # dist.destroy_process_group()
    rpc.shutdown()  # 关闭 RPC

def main():
    """主函数"""
    logging.info("[Main] 启动分布式系统")
    world_size = sum(count for _, count in PROCESS_TYPES)  # 根据定义的进程类型计算 world_size
    logging.info(f"[Main] world_size = {world_size}，进程类型分布: {PROCESS_TYPES}")
    
    if world_size < 2:
        raise ValueError("需要至少 2 个进程来运行此程序。")
    
    mp.spawn(init_process, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
