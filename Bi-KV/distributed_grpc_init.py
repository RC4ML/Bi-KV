from datetime import timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TORCH_CUDA_ARCH_LIST.*")
import os
import time
import grpc
import torch.multiprocessing as mp
import torch.distributed as dist
from protos import TaskInfo_pb2, TaskInfo_pb2_grpc
from Scheduler.LLMScheduler import LLMScheduler
from DistributedStorage.CacheCoordinator import CacheCoordinator
from DistributedStorage.kvcache import KVCache
from inputGenerator.inputGenerator import LLMInput
from Worker.Worker import Worker
import logging
from config import *
from rpc_def import *
from Remote.remote_call import call_remote_method
from concurrent import futures
from network import *
import yaml
import json
import pickle

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

args.model_code = 'llm'
args.llm_base_model = "/share/nfs/models/Llama-2-7b-hf"
args.llm_base_tokenizer = "/share/nfs/models/Llama-2-7b-hf"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("distributed_system.log"),
        logging.StreamHandler()
    ]
)

def load_config(file_path):
    with open(file_path, 'r') as file:
        yaml_config = yaml.safe_load(file)
    return yaml_config

def init_backend(rank, world_size, process_type, type_index, timeout=4000000):
    local_ip = get_local_ip()
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size,timeout = timedelta(seconds=timeout))
    dist.barrier()
    logging.info(f"Process ID: {process_type} {type_index}, Rank: {rank}, IP: {local_ip}")

def init_process(rank, world_size, yaml_config):
    slots = yaml_config['grpc']['slots']
    rank_to_ip = set_rank_to_ip(slots)
    process_type, type_index = get_process_info(rank)
    rank_map = generate_rank_map(world_size)
    master_addr = os.environ['MASTER_ADDR']
    
    master_port = int(os.environ['MASTER_PORT']) # 50500
    if process_type == 'scheduler':
        timeout = 100000
    else:
        timeout = 36000

    init_backend(rank, world_size, process_type, type_index, timeout=timeout)

    if process_type == 'LLMScheduler':
        dataset_code = yaml_config['general']['dataset_code']
        max_batch_token = yaml_config['scheduler']['max_batch_token']
        user_expand_ratio = yaml_config['input_generator']['user_history_expand_ratio']
        recommendation_size = yaml_config['input_generator']['recommendation_size']
        iter_round = yaml_config['scheduler']['iter_round']
        batch_size = yaml_config['scheduler']['batch_size']
        input_batch_size = yaml_config['scheduler']['input_batch_size']
        hack_option_dict = {0:"compete", 1:"User History First", 2:"Item First"}
        hack_option = None # hack_option_dict[0]

        with open(f'/share/nfs/wsh/Bi-KV/Bi-KV/data/{args.dataset_code}/prepare_cache_data_user_all.json', 'r') as f:
            load_data_user = json.load(f)

        with open(f'/share/nfs/wsh/Bi-KV/Bi-KV/data/{args.dataset_code}/prepare_cache_data_item_all.json', 'r') as f:
            load_item_user = json.load(f)

        timestamp_map_path = f'/share/nfs/wsh/Bi-KV/Bi-KV/data/{args.dataset_code}/timestep_map.json'
        with open(timestamp_map_path, 'r') as f:
            time_step_map = json.load(f)

        item_access_path = f"/share/nfs/wsh/Bi-KV/Bi-KV/data/{args.dataset_code}/user_candidate.pickle"
        with open(item_access_path, 'rb') as pickle_file:
            user_candidate_dict = pickle.load(pickle_file)

        loaded_data = {"user": load_data_user, "item": load_item_user, "time_step_map": time_step_map, "user_candidate_dict": user_candidate_dict}

        scheduler = LLMScheduler(world_size=world_size, master_port=master_port, rank_to_ip=rank_to_ip,max_batch_token=max_batch_token,batchsize=input_batch_size)
        input_generator = LLMInput(recommendation_size, 5, args,user_expand_ratio,False)
        scheduler.set_prompt_generator(input_generator)
        dist.barrier()
        CacheCoordinator_addr = f"{master_addr}:{master_port + rank_map['CacheCoordinator'][0]}"
        channel = grpc.insecure_channel(CacheCoordinator_addr)
        stub = TaskInfo_pb2_grpc.CacheCoordinatorServiceStub(channel)
        fut = stub.StartProcessRequest.future(TaskInfo_pb2.StartRequest(msg='start'))
        logging.info(f"Cache Size: {yaml_config['kv_cache']['cache_size']} Page Size: {yaml_config['kv_cache']['page_size']}")
        logging.info(f"Worker Buffer Size: {yaml_config['worker']['cache_size']} Page Size: {yaml_config['worker']['page_size']}")
        logging.info(f"Max Batch Token: {max_batch_token}")
        logging.info(f"User History Expand Ratio: {user_expand_ratio}")
        logging.info(f"Recommendation Size: {recommendation_size}")
        logging.info(f"Iter Round: {iter_round} Batch Size: {batch_size}")
        logging.info(f"Test Hack Option: {hack_option}")
        logging.info("Start Testing")
        time1 = time.time()
        # time_step_map = None
        scheduler.start_test(iter_round,batch_size,loaded_data,hack_option=hack_option)
        fut.result()
        channel.close()
        time2 = time.time()
        logging.info(f"Test Time cost: {time2 - time1}")

    if process_type == 'CacheCoordinator':
        dist.barrier()
        pass

    if process_type == 'Worker':
        port = master_port + rank
        cache_size = yaml_config['worker']['cache_size']
        page_size = yaml_config['worker']['page_size']
        max_workers = yaml_config['worker']['max_workers']
        rank_to_ip_rdma = yaml_config['distributed']['rank_to_ip_rdma']
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        worker = Worker(rank, master_port, cache_size, page_size, rank_map['CacheCoordinator'][0], rank_to_ip, rank_to_ip_rdma, server)
        TaskInfo_pb2_grpc.add_InferWorkerServiceServicer_to_server(
            worker, server
        )
        server.add_insecure_port(f'{rank_to_ip[rank]}:{port}')
        server.start()
        # worker.start_rdma()
        dist.barrier()
        server.wait_for_termination()

    if process_type == 'KVCache':
        port = master_port + rank
        cache_size = yaml_config['kv_cache']['cache_size']
        page_size = yaml_config['kv_cache']['page_size']
        max_workers = yaml_config['kv_cache']['max_workers']
        rank_to_ip_rdma = yaml_config['distributed']['rank_to_ip_rdma']
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        cache = KVCache(rank, cache_size, page_size, master_port, rank_to_ip,rank_to_ip_rdma, server)
        TaskInfo_pb2_grpc.add_KVCacheServiceServicer_to_server(
           cache, server
        )
        server.add_insecure_port(f'{rank_to_ip[rank]}:{port}')
        server.start()
        # cache.start_rdma()
        dist.barrier()
        server.wait_for_termination()

    dist.barrier()
    dist.destroy_process_group()

def main():
    yaml_config = load_config("../config.yml")
    dataset_code = yaml_config['general']['dataset_code']
    args.dataset_code = dataset_code
    args.llm_retrieved_path = f"/share/nfs/sunjie/{dataset_code}"
    init_network()
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ.get('RANK', os.environ.get('OMPI_COMM_WORLD_RANK', -1)))
    if WORLD_RANK == 0:
        logging.info(f"Bi-KV 启动分布式系统")

    rank = WORLD_RANK
    world_size = WORLD_SIZE
    assert world_size == sum(count for _, count in PROCESS_TYPES)
    assert world_size >= 2
    init_process(rank, world_size, yaml_config)

if __name__ == "__main__":
    main()