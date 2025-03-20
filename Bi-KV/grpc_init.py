import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TORCH_CUDA_ARCH_LIST.*")
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用所有GPU
import grpc
import yaml

import torch.multiprocessing as mp
import torch.distributed as dist
from protos import TaskInfo_pb2,TaskInfo_pb2_grpc

from Scheduler.LLMScheduler import LLMScheduler
from DistributedStorage.CacheCoordinator import CacheCoordinator
from DistributedStorage.kvcache import KVCache
from inputGenerator.inputGenerator import LLMInput
from Worker.Worker import Worker
import warnings
import logging
from config import *
from rpc_def import PROCESS_TYPES, KVCACHE_NUM, WORKER_NUM, get_process_info, generate_rank_map
from Remote.remote_call import call_remote_method

from concurrent import futures


warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

args.model_code = 'llm'
# args.llm_retrieved_path =  "/share/nfs/sunjie/games"
args.llm_retrieved_path = "/data/testmodel/LlamaRec/experiments/lru/games"
args.dataset_code = "games"

def load_config(file_path):
    with open(file_path, 'r') as file:
        yaml_config = yaml.safe_load(file)
    return yaml_config

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("distributed_system.log"),
        logging.StreamHandler()
    ]
)

def init_backend(rank, world_size, process_type, type_index, yaml_config):
    os.environ["MASTER_ADDR"] = yaml_config['distributed']['master_addr']
    os.environ["MASTER_PORT"] = yaml_config['distributed']['master_port']
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    
    logging.info(f"[init_backend] 初始化进程类型: {process_type}{type_index}, Rank: {rank}")

def init_process(rank, world_size,yaml_config):
    process_type, type_index = get_process_info(rank)
    rank_map = generate_rank_map(world_size)
    master_port = yaml_config['grpc']['master_port']
    init_backend(rank, world_size, process_type, type_index, yaml_config)
    dist.barrier()
    
    if process_type == 'LLMScheduler':
        logging.info(f"[init_process][Rank {rank}] 初始化 LLMScheduler")
        scheduler = LLMScheduler(world_size=world_size,master_port=master_port)
        # scheduler.test_write_cache()
        input_generator = LLMInput(20,5,args)
        # input_generator.set_random('random')
        scheduler.set_prompt_generator(input_generator)
        CacheCoordinator_addr = f"localhost:{master_port+rank_map['CacheCoordinator'][0]}"
        channel = grpc.insecure_channel(CacheCoordinator_addr)
        stub = TaskInfo_pb2_grpc.CacheCoordinatorServiceStub(channel)
        fut = stub.StartProcessRequest.future(TaskInfo_pb2.StartRequest(msg='start'))
            # stub.StartProcess(TaskInfo_pb2.StartRequest(msg='start'))
        logging.info("开始测试")
        time1 = time.time()
        scheduler.start(10,256)
        time2 = time.time()
        print(f"Test Time cost: {time2-time1}")
        fut.result()
        channel.close()

    if process_type == 'CacheCoordinator':
        # port = master_port + rank
        # server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        # # 需要传入cache和worker的地址
        # TaskInfo_pb2_grpc.add_CacheCoordinatorServiceServicer_to_server(
        #     CacheCoordinator(rank,master_port,rank_map['KVCache'],rank_map['Worker']), server
        # )
        # server.add_insecure_port(f'[::]:{port}')
        # server.start()
        # print(f"CacheCoordinator started on port {port}")
        # server.wait_for_termination()
        pass

    if process_type == 'Worker':
        time.sleep(5) # wait shared memory
        port = master_port + rank
        cache_size = yaml_config['worker']['cache_size']
        page_size = yaml_config['worker']['page_size']
        max_workers = yaml_config['worker']['max_workers']
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        worker = Worker(rank,master_port,cache_size,page_size,rank_map['CacheCoordinator'][0],server)
        TaskInfo_pb2_grpc.add_InferWorkerServiceServicer_to_server(
            worker, server
        )
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        print(f"Worker[{rank}] started on port {port}")
        server.wait_for_termination()

    if process_type == 'KVCache':
        cache_size = yaml_config['kv_cache']['cache_size']
        page_size = yaml_config['kv_cache']['page_size']
        max_workers = yaml_config['kv_cache']['max_workers']
        port = master_port + rank
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        kv_cache = KVCache(rank, cache_size, page_size, master_port, server)
        TaskInfo_pb2_grpc.add_KVCacheServiceServicer_to_server(
            kv_cache, server
        )
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        print(f"KVCache[{rank}] started on port {port}")
        server.wait_for_termination()

    dist.barrier()
    dist.destroy_process_group()

def main():
    yaml_config = load_config("../config.yml")
    logging.info("[Main] 启动分布式系统")
    world_size = sum(count for _, count in yaml_config['process_types'].items())
    logging.info(f"[Main] world_size = {world_size}，进程类型分布: {PROCESS_TYPES}")
    
    if world_size < 2:
        raise ValueError("需要至少 2 个进程来运行此程序。")
    
    mp.spawn(init_process, args=(world_size,yaml_config,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
