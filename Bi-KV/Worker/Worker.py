from ast import Dict, List
import json
import time

from protos import TaskInfo_pb2,TaskInfo_pb2_grpc
import grpc

import torch.distributed.rpc as rpc
import torch.distributed as dist

from rpc_def import *
from DistributedStorage.CacheCoordinator import CacheCoordinator
from DistributedStorage.PageManager import PageManager
from DistributedStorage.Signals import SIGNAL_RECV,CACHE_MISS
from Remote.remote_call import call_remote_method
from Utils.utils import now_time
from Utils.channelpool import ChannelPool

# from rdma_transport import RDMAEndpoint
from rdma_onesided_transport import RDMAOneSidedEndpoint
import ipc_service

import torch
from config import *
from Model.qwen2 import process_task_info, prepare_attention_meta, Qwen2ForCausalLM, AttentionMetadata, token_shape, model_params
from transformers import Qwen2Config
from vllm.config import CacheConfig

from datetime import datetime
import time
import os
import logging


torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

class Worker(TaskInfo_pb2_grpc.InferWorkerServiceServicer):
    def __init__(self, rank, master_port,cache_size,page_size, coordinator_rank, rank_to_ip, rank_to_ip_rdma, server):
        self.rank = rank
        self.master_port = master_port
        self.worker_index=int(rank/2) -1
        self.coordinator_rank = coordinator_rank
        master_addr = rank_to_ip[1] # rank = 1 for coordinator
        self.server = server
        self.rank_to_ip_grpc = rank_to_ip
        self.rank_to_ip_rdma = rank_to_ip_rdma
        self.cache_coordinator_address = f"{master_addr}:{master_port+coordinator_rank}"
        self.gpu_index = 0 # self.worker_index # NOTE: set to rank in a single machine
        self.device = torch.device(f"cuda:{self.gpu_index}")
        # key item id value(start_pos,offset) req_id?
        # 多个req_id并发的情况？
        self.buffer_size = cache_size
        self.page_size = page_size
        # PageManager会不会遇到并发？？？
        self.page_manager = PageManager(cache_size=self.buffer_size, page_size=self.page_size)
        self.compute_buffer = torch.full(
            (self.buffer_size,) + token_shape, 
            self.rank,
            device='cpu',
            dtype=torch.float16
        )
        self.start_pos = 0
        self.cache_miss_dict = {}
        self.server = server
        
        # initialize model and parameters for inference
        intermediate_size = 8960
        head_dim = 128
        kv_cache_block_size = 16
        max_kv_cache_blocks = 10240        
        self.local_kv_cache_block_size = kv_cache_block_size
        self.local_max_kv_cache_blocks = max_kv_cache_blocks
        self.local_kvcache = [
            torch.randn(
            self.local_max_kv_cache_blocks * self.local_kv_cache_block_size, 2, model_params['num_kv_heads'], head_dim, dtype=torch.float16, device=self.device
            ) for _ in range(model_params['num_layers'])
        ]
        print(self.local_kvcache[0].shape)
        self.cache_config = CacheConfig(kv_cache_block_size, 1.0, 1, "auto")
        self.model_config = Qwen2Config(hidden_size = model_params['num_q_heads']*head_dim,
                                        intermediate_size = intermediate_size,
                                        num_hidden_layers = model_params['num_layers'],
                                        num_attention_heads = model_params['num_q_heads'],
                                        num_key_value_heads = model_params['num_kv_heads'])
        torch.set_default_dtype(torch.float16)
        self.model = Qwen2ForCausalLM(self.device, self.model_config, self.cache_config).to(self.device)

        # 初始化消费者端共享内存
        self.shm_name = f"/sj_kv_cache_{self.worker_index}"
        self._init_shared_memory()
        self.channelpool = ChannelPool()
 
    def _init_shared_memory(self):
        """初始化CUDA共享内存区域"""
        device_id = self.worker_index
        try:
            # 先尝试清理可能存在的残留共享内存
            ipc_service.consumer_cleanup()  
        except:
            pass
        buffer_size = self.compute_buffer.element_size() * self.compute_buffer.nelement()
        print(f"buffer_size:{buffer_size/(1024**2)}MB")
        # 初始化生产者端共享内存
        ipc_service.consumer_init(device_id, self.shm_name.encode(), buffer_size*5)
        
    def __del__(self):
        print(f"Worker {self.rank} destroyed at {time.time()}") 
    
    def start_rdma(self):        
        self.ep = {}
        max_retries = 100  # 最大重试次数
        retry_delay = 0.5  # 每次重试的间隔时间（秒）
        self.buffer_size = 1024*1024*1024

        for cid in range(KVCACHE_NUM):
            retries = 0
            self.ep[cid*2+WORKER_offset] = RDMAOneSidedEndpoint(self.rank_to_ip_rdma[cid*2+WORKER_offset], str(self.master_port+10), "client")

            while retries < max_retries:
                try:
                    if self.ep[cid*2+WORKER_offset].connect_client(rank=self.rank, cpu_size=self.buffer_size, gpu_size=0, hugepage=False) == 0:
                        # logging.info(f"Client {self.rank} connection {self.rank_to_ip_rdma[wid*2+WORKER_offset]}:{self.master_port+10} success!")
                        break  # 连接成功，退出重试循环
                    else:
                        # logging.info(f"Client {self.rank} connection {self.rank_to_ip_rdma[wid*2+WORKER_offset]}:{self.master_port+10} attempt {retries + 1} failed! Retrying in {retry_delay} seconds...")
                        retries += 1
                        time.sleep(retry_delay)
                except Exception as e:
                    # logging.info(f"Client {self.rank} connection {self.rank_to_ip_rdma[wid*2+WORKER_offset]}:{self.master_port+10} attempt {retries + 1} failed with error: {e}. Retrying in {retry_delay} seconds...")
                    retries += 1
                    time.sleep(retry_delay)
            
            # 如果重试次数用尽仍未成功，则断言失败
            if retries == max_retries:
                logging.info(f"Client {self.rank} connection failed after {max_retries} attempts!")
                assert 0

        logging.info(f"RDMA client started at {self.rank_to_ip_rdma[self.rank]}!")

    def ReceiveTasksFromScheduler(self, request, context):
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv taskinfo length:{len(request.tasks)} from scheduler")
        for task in request.tasks:
            item_id = task.id
            if item_id == -1:
                continue
            self.page_manager.load_item(item_id, task.token_num)
            self.page_manager.set_protected(item_id)
        self.forward_with_computation_grpc(request.tasks)
        return TaskInfo_pb2.Empty()
    
    def StartWriteCacheData(self, request, context):
        self.preprare_send_data_grpc(request.tasks)
        return TaskInfo_pb2.Empty()

    def receive_task_info_batch(self, task_info_list):
        # 按照req_id分组
        self.receive_task_info(task_info_list)
        # task_info_dict = {}
        # for i in task_info_list:
        #     req_id = i['request_id']
        #     if req_id not in task_info_dict:
        #         task_info_dict[req_id] = []
        #     task_info_dict[req_id].append(i)
        # for i in task_info_dict.values():
        #     self.receive_task_info(i)

    def receive_kvcache_data_batch(self, combined_task_info):
        cache_worker = combined_task_info.cache_worker
        src_rank = 2*cache_worker + KVCACHE_offset
        token_num = combined_task_info.token_num
        recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
        )
        if self.worker_index == cache_worker:
            # 从共享内存接收CUDA张量
            start_read=time.time()
            # recv_tensor=ipc_service.consumer_receive()
            end_read=time.time()
            
            total_bytes = recv_tensor.numel() * recv_tensor.element_size()  # 正确计算总字节
            time_diff = end_read - start_read
            throughput = total_bytes / time_diff / 1e9  # 转换为GB/s
            print(f"[ipc_service.consumer_receive]shared once time: {time_diff}s, torch.size{recv_tensor.size()},total_bytes:{total_bytes/(1024**2)}MB, "
                     f"throughput: {throughput} GB/s\n")
        else: 
            start_recv=time.time()
            # self.ep.post_send_by_rank(src_rank, token_num * 128 * 8 * 28)
            # self.ep.poll_completion_by_rank(src_rank)
            end_recv=time.time()
            #计算总数据量（字节）
            total_bytes = recv_tensor.numel() * recv_tensor.element_size()  # 正确计算总字节
            time_diff = end_recv - start_recv
            throughput = total_bytes / time_diff / 1e9  # 转换为GB/s
            # print(f"[dist.recv]send once time: {time_diff}s, torch.size{recv_tensor.size()},total_bytes:{total_bytes/(1024**2)}MB, "
            #     f"throughput: {throughput} GB/s\n")
        
    def send_kvcache_data_batch(self, combined_task_info):
        dst_rank = 2*combined_task_info.cache_worker + KVCACHE_offset
        token_num = combined_task_info.token_num
        id_pair_list = combined_task_info.id_token_pair
        # 计算总 token 数并预分配索引
        total_token_num = sum(id_pair.token_num for id_pair in id_pair_list)
        buffer_indices = torch.empty(total_token_num, dtype=torch.long)
        send_indices = torch.empty(total_token_num, dtype=torch.long)

        # 第一步：收集索引
        # buffer_pos = 0
        # send_pos = 0
        # for id_pair in id_pair_list:
        #     i = id_pair.id
        #     token_num = id_pair.token_num
            
        #     if i not in self.page_manager.get_loaded_lists():
        #         print(f"[Worker][RANK {self.rank}][{time.time()}] Error: id {i} not in page manager")
            
        #     page_set = self.page_manager.access_item(i)
            
        #     # 生成索引
        #     for idx, page in enumerate(page_set):
        #         if idx == len(page_set) - 1:
        #             size = token_num % self.page_size if token_num % self.page_size != 0 else self.page_size
        #             buffer_indices[buffer_pos:buffer_pos + size] = torch.arange(
        #                 page * self.page_size,
        #                 page * self.page_size + size
        #             )
        #             send_indices[send_pos:send_pos + size] = torch.arange(
        #                 send_pos,
        #                 send_pos + size
        #             )
        #             buffer_pos += size
        #             send_pos += size
        #         else:
        #             buffer_indices[buffer_pos:buffer_pos + self.page_size] = torch.arange(
        #                 page * self.page_size,
        #                 (page + 1) * self.page_size
        #             )
        #             send_indices[send_pos:send_pos + self.page_size] = torch.arange(
        #                 send_pos,
        #                 send_pos + self.page_size
        #             )
        #             buffer_pos += self.page_size
        #             send_pos += self.page_size

        # # 第二步：一次性创建并填充 send_tensor
        send_tensor = torch.empty(
            (total_token_num,) + token_shape,
            dtype=torch.float16
        )
        # send_tensor[send_indices] = self.compute_buffer[buffer_indices]

        # 第三步：移除保护
        for id_pair in id_pair_list:
            self.page_manager.remove_protected(id_pair.id)
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 长度={token_num}")
        # dist.send(tensor=send_tensor, dst=dst_rank)
        # self.ep.post_send_by_rank(dst_rank, token_num * 128 * 8 * 28)
        # self.ep.poll_completion_by_rank(dst_rank)
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 长度={token_num}")
            
    def forward_with_computation_grpc(self, tasks):
        tasks_list = tasks
        tasks = TaskInfo_pb2.TaskInfoList(tasks=tasks)
        # with grpc.insecure_channel(self.cache_coordinator_address) as channel:
        channel = self.channelpool.get_channel(self.cache_coordinator_address)
        stub = TaskInfo_pb2_grpc.CacheCoordinatorServiceStub(channel)
        stub.ReceiveTasksFromInferWorker(tasks)  # 直接转发整个 TaskInfoList
        if DEBUG:
            print(f"{now_time()}[Worker {self.rank}] try to poll_batch")
        # with grpc.insecure_channel(self.cache_coordinator_address) as channel:
        channel = self.channelpool.get_channel(self.cache_coordinator_address)
        stub = TaskInfo_pb2_grpc.CacheCoordinatorServiceStub(channel)
        # stub.PollBatchFromInferWorker(tasks)
        time1 = time.time()
        future_call_poll = stub.PollBatchFromInferWorker.future(tasks)  # 直接转发整个 TaskInfoList
        if DEBUG:
            print(f"[Worker.forward_with_computation][RANK {self.rank}] finsh CacheCoordinator.poll_batch")
        cache_miss_dict_data = future_call_poll.result()
        time2 = time.time()
        # logging.info(f"[Worker][RANK {self.rank}] poll time {time2-time1}s")
        cache_miss_dict = json.loads(cache_miss_dict_data.msg)
        # logging.info(f"[Worker.forward_with_computation][RANK {self.rank}] cache_miss_dict: {cache_miss_dict}")
        # cache_miss_dict = future_call_poll.result()
        for req_id in cache_miss_dict:
            self.cache_miss_dict[req_id] = cache_miss_dict[req_id]
        # ## start model inference
        time3 = time.time()
        queried_task_info_list = process_task_info(tasks_list, cache_miss_dict)
        attn_metadata, cached_tokens = prepare_attention_meta(queried_task_info_list, self.local_kv_cache_block_size, self.local_max_kv_cache_blocks, self.device)
        input_ids = torch.zeros(attn_metadata.nnz_qo, dtype=torch.int32, device=self.device)
        positions = torch.arange(attn_metadata.nnz_qo, dtype=torch.int64, device=self.device)
        time4 = time.time()
        # print(f"shape {input_ids.shape} {cached_tokens}")
        output = self.model(input_ids, positions, self.local_kvcache, attn_metadata)    
        time5 = time.time()
        logging.info(f"worker {self.worker_index}, read kv cache time {time3-time1}s, compute time: {time5-time3}s")
    
    def preprare_send_data_grpc(self, task_info_list):
        send_task_list = []
        hit_counter = 0
        length_counter = 0
        
        hit_counter_user = 0
        length_counter_user = 0
        hit_counter_item = 0
        length_counter_item = 0
        
        for task_info in task_info_list:
            item_id = task_info.id
            request_id = task_info.request_id
            cache_miss_dict = self.cache_miss_dict.get(str(request_id),{})
            hit_counter += sum(cache_miss_dict.values())
            length_counter += len(cache_miss_dict)
            if task_info.type == 'user cache':
                length_counter_user += len(cache_miss_dict)
                hit_counter_user += sum(cache_miss_dict.values())
            elif task_info.type == 'item cache':
                length_counter_item += len(cache_miss_dict)
                hit_counter_item += sum(cache_miss_dict.values())
            if cache_miss_dict.get(str(item_id)) == CACHE_MISS:
                # print(f"[Worker][RANK {self.rank}] Cache miss detected")
                task_info.task_type = SIGNAL_RECV
                send_task_list.append(task_info)
            else:
                if item_id in self.page_manager.get_loaded_lists():
                    self.page_manager.remove_protected(item_id)
        # TODO 适配
        # logging.info(f"length counter {hit_counter}/{length_counter} ")
        hit_rate = hit_counter/length_counter
        user_hit_rate = hit_counter_user/length_counter_user if length_counter_user != 0 else 0
        item_hit_rate = hit_counter_item/length_counter_item if length_counter_item != 0 else 0
        # if hit_rate > 0.7 and DEBUG:
        # logging.info(f"[Worker {self.rank}] User Hit rate: {user_hit_rate} Item Hit rate: {item_hit_rate} Total Hit rate: {hit_rate}") 
        # print(f"[Worker][RANK {self.rank}] Sending data to kvcache")
        if len(send_task_list)>0:
            send_task_list_gprc = TaskInfo_pb2.TaskInfoList(tasks=send_task_list)
            # with grpc.insecure_channel(self.cache_coordinator_address) as channel:
            channel = self.channelpool.get_channel(self.cache_coordinator_address)
            stub = TaskInfo_pb2_grpc.CacheCoordinatorServiceStub(channel)
            stub.ReceiveTasksFromInferWorker(send_task_list_gprc)

    def RecvKVCacheData(self, request, context):
        task_info = request
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv kvcache data {task_info} from kvcache")
        self.receive_kvcache_data_batch(task_info)
        return TaskInfo_pb2.Empty()
    
    def SendKVCacheData(self, request, context):
        task_info = request
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Sending data to kvcache")
        self.send_kvcache_data_batch(task_info)
        return TaskInfo_pb2.Empty()
    
    def ShutDown(self, request, context):
        # 在这里处理关闭逻辑
        self.server.stop(0)
        return TaskInfo_pb2.Empty()
