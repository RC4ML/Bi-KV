from ast import Dict, List
import json
import time

from rpc_def import KVCACHE_NUM ,WORKER_NUM
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
import ipc_service


import torch
from config import *
from Model.qwen2 import process_task_info, prepare_attention_meta, Qwen2ForCausalLM, AttentionMetadata, token_shape, model_params
from transformers import Qwen2Config
from vllm.config import CacheConfig

import time

class Worker(TaskInfo_pb2_grpc.InferWorkerServiceServicer):
    def __init__(self, rank,master_port:int,cache_size,page_size, coordinator_rank = None, server=None):
        self.rank = rank
        self.worker_index=int(rank/2) -1
        self.coordinator_rank = coordinator_rank
        self.cache_coordinator_address = f"localhost:{master_port+coordinator_rank}"
        self.gpu_index = self.worker_index # NOTE: set to rank in a single machine
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
        
        ## initialize model and parameters for inference
        # intermediate_size = 8960
        # head_dim = 128
        # kv_cache_block_size = 16
        # max_kv_cache_blocks = 10240        
        # self.local_kv_cache_block_size = kv_cache_block_size
        # self.local_max_kv_cache_blocks = max_kv_cache_blocks
        # self.local_kvcache = [
        #     torch.randn(
        #     self.local_max_kv_cache_blocks * self.local_kv_cache_block_size, 2, model_params['num_kv_heads'], head_dim, dtype=torch.float16, device=self.device
        #     ) for _ in range(model_params['num_layers'])
        # ]
        # print(self.local_kvcache[0].shape)
        # self.cache_config = CacheConfig(kv_cache_block_size, 1.0, 1, "auto")
        # self.model_config = Qwen2Config(hidden_size = model_params['num_q_heads']*head_dim,
        #                                 intermediate_size = intermediate_size,
        #                                 num_hidden_layers = model_params['num_layers'],
        #                                 num_attention_heads = model_params['num_q_heads'],
        #                                 num_key_value_heads = model_params['num_kv_heads'])
        # torch.set_default_dtype(torch.float16)
        # self.model = Qwen2ForCausalLM(self.device, self.model_config, self.cache_config).to(self.device)
        # 初始化消费者端共享内存
        self.shm_name = f"/worker_buffer_{self.worker_index}"
        self._init_shared_memory()

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

    def forward(self, task_info_list:List):
        coordinator_owner = self.coordinator_rank.owner()
        req_id = task_info_list[0]['request_id']
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Add {len(task_info_list)} requests to coordinator")
        rpc.rpc_sync(to=coordinator_owner, 
                         func=call_remote_method, 
                         args=(CacheCoordinator.add_requests,self.coordinator_rank, 
                               task_info_list))
        finished_signal = False
        cache_miss_dict = {'0':-1}
        while not finished_signal and (-1 in cache_miss_dict.values() or cache_miss_dict == {}):
            if DEBUG:
                print(f"[Worker][RANK {self.rank}] Poll requests...")
            future_call_poll = rpc.rpc_async(to=coordinator_owner,func=call_remote_method, 
                                         args=(CacheCoordinator.poll,self.coordinator_rank,task_info_list))
            res = future_call_poll.wait()
            if DEBUG:
                print(f"[Worker][RANK {self.rank}] Poll result: {res} Task info list: {[task['id'] for task in task_info_list]}")
            finished_signal = res[0]
            cache_miss_dict = res[1]
            if DEBUG:
                print(f"[Worker][RANK {self.rank}] Cache miss dict: {cache_miss_dict}")
            if finished_signal and -1 not in cache_miss_dict.values():
                if DEBUG:
                    print(f"[Worker][RANK {self.rank}] Requests finished")
            else:
                if DEBUG:
                    print(f"[Worker][RANK {self.rank}] Requests are still being processed...")
                time.sleep(5)
            if 0 in cache_miss_dict.values():
                pass
                # print(f"[Worker][RANK {self.rank}] Cache miss detected")
            self.cache_miss_dict[req_id] = cache_miss_dict
        # print(f"[Worker][RANK {self.rank}] Moving compute buffer to device {self.gpu_index}...")
        # self.compute_buffer.to(self.device)

    def forward_with_computation(self, task_info_list:List):
        # 这里的task_info同样有着一样的req_id
        time1 = time.time()
        coordinator_owner = self.coordinator_rank.owner()
        if DEBUG:
            print(f"[Worker.forward_with_computation][RANK {self.rank}] Add {len(task_info_list)} requests to coordinator")
        rpc.rpc_sync(to=coordinator_owner, 
                         func=call_remote_method, 
                         args=(CacheCoordinator.add_requests,self.coordinator_rank, 
                               task_info_list))
        time2 = time.time()
        
        future_call_poll = rpc.rpc_async(to=coordinator_owner,func=call_remote_method, 
                            args=(CacheCoordinator.poll_batch,self.coordinator_rank,task_info_list))
        if DEBUG:
            print(f"[Worker.forward_with_computation][RANK {self.rank}] finsh CacheCoordinator.poll_batch")
        # cache_miss_dict是一个嵌套字典，第一层是req_id，第二层是item_id
        cache_miss_dict = future_call_poll.wait()
        for req_id in cache_miss_dict:
            self.cache_miss_dict[req_id] = cache_miss_dict[req_id]

        # ## start model inference
        # time3 = time.time()
        # queried_task_info_list = process_task_info(task_info_list)
        # attn_metadata, cached_tokens = prepare_attention_meta(queried_task_info_list, self.local_kv_cache_block_size, self.local_max_kv_cache_blocks, self.device)
        # input_ids = torch.zeros(attn_metadata.nnz_qo, dtype=torch.int32, device=self.device)
        # positions = torch.arange(attn_metadata.nnz_qo, dtype=torch.int64, device=self.device)
        # time4 = time.time()
        # # print(f"shape {input_ids.shape} {cached_tokens}")
        # output = self.model(input_ids, positions, self.local_kvcache, attn_metadata)    
        # time5 = time.time()
        # print(f"worker {self.worker_index}, read kv cache time {time3-time2}s, compute time: {time5-time3}s, 100Gbps network time: {(cached_tokens*model_params['num_kv_heads']*model_params['head_size']*model_params['num_layers']*2*2)/(12*1000*1000*1000)}s")

    def receive_task_info(self, task_info_list):
        # 此时的task_info_list是一个request的所有task，req_id相同
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv taskinfo length:{len(task_info_list)} from scheduler")
        for task_info in task_info_list:
            item_id = task_info['id']
            if item_id == -1:
                continue
            self.page_manager.load_item(item_id, task_info['token_num'])
            self.page_manager.set_protected(item_id)
        self.forward_with_computation(task_info_list)
        if DEBUG:
            print(f"[Worker.receive_task_info][RANK {self.rank}] Sending data to kvcache")
        self.preprare_send_data(task_info_list)
        # print(f"[Worker][RANK {self.rank}]finish receive_task_info")

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
        self.preprare_send_data_grpc(request.tasks)
        return TaskInfo_pb2.Empty()


    def receive_task_info_batch(self, task_info_list):
        # 按照req_id分组
        task_info_dict = {}
        for i in task_info_list:
            req_id = i['request_id']
            if req_id not in task_info_dict:
                task_info_dict[req_id] = []
            task_info_dict[req_id].append(i)
        for i in task_info_dict.values():
            self.receive_task_info(i)
        if DEBUG:
            print(f"[Worker][RANK {self.rank}]finish receive_task_info_batch")

    def receive_kvcache_data(self, task_info):
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv kvcache data {task_info} from kvcache")
        self.write_compute_buffer(task_info)

    def receive_kvcache_data_batch(self, combined_task_info):
        cache_worker = combined_task_info.cache_worker
        src_rank = 2*cache_worker + KVCACHE_offset
        token_num = combined_task_info.token_num
        if self.worker_index == cache_worker:
            # print(f"[Worker][RANK {self.rank}] start get shared memory")
            # 从共享内存接收CUDA张量
            start_read=time.time()
            recv_tensor=ipc_service.consumer_receive()
            end_read=time.time()
            
            # 将张量复制到CPU
            # recv_tensor = cuda_tensor.cpu()
            # print(f"shared{recv_tensor.size()}")
            # del cuda_tensor
            # 显式释放显存
            # del cuda_tensor
            # torch.cuda.empty_cache()  # 可选但建议添加
           # 计算总数据量（字节）
            total_bytes = recv_tensor.numel() * recv_tensor.element_size()  # 正确计算总字节
            time_diff = end_read - start_read
            throughput = total_bytes / time_diff / 1e9  # 转换为GB/s
            print(f"[ipc_service.consumer_receive]shared once time: {time_diff}s, torch.size{recv_tensor.size()},total_bytes:{total_bytes/(1024**2)}MB, "
                     f"throughput: {throughput} GB/s\n")
            # with open(f'shared_log_rank{self.rank}.txt', 'a+') as f:
            #     f.write(f"[ipc_service.consumer_receive]shared once time: {time_diff}s, torch.size{recv_tensor.size()},total_bytes:{total_bytes/(1024**2)}MB, "
            #         f"throughput: {throughput} GB/s\n")
        else: 
            #print("skip send")
            start_recv=time.time()
            recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
            )
            dist.recv(tensor=recv_tensor, src=src_rank)
            end_recv=time.time()
            # #计算总数据量（字节）
            # total_bytes = recv_tensor.numel() * recv_tensor.element_size()  # 正确计算总字节
            # time_diff = end_recv - start_recv
            # throughput = total_bytes / time_diff / 1e9  # 转换为GB/s
            # with open(f'send_log_rank{self.rank}.txt', 'a+') as f:
            #     f.write(f"[dist.recv]send once time: {time_diff}s, torch.size{recv_tensor.size()},total_bytes:{total_bytes/(1024**2)}MB, "
            #     f"throughput: {throughput} GB/s\n")
            # print(f"send{recv_tensor.size()}")
        # # 计算总大小并预分配索引 tensor
        # total_size = sum(id_pair.token_num for id_pair in combined_task_info.id_token_pair)
        # indices = torch.empty(total_size, dtype=torch.long)
        # buffer_indices = torch.empty(total_size, dtype=torch.long)

        # # 第一步：收集索引
        # start_pos = 0
        # buffer_pos = 0
        # for id_pair in combined_task_info.id_token_pair:
        #     item_id = id_pair.id
        #     offset = id_pair.token_num
            
        #     # 管理 page
        #     if item_id not in self.page_manager.get_loaded_lists():
        #         page_set, _ = self.page_manager.load_item(item_id, offset)
        #     else:
        #         page_set = self.page_manager.access_item(item_id)
            
        #     # 生成索引
        #     item_size = offset
        #     indices[start_pos:start_pos + item_size] = torch.arange(start_pos, start_pos + item_size)
            
        #     # 生成 buffer 对应的索引
        #     for idx, page in enumerate(page_set):
        #         if idx == len(page_set) - 1:
        #             size = offset % self.page_size if offset % self.page_size != 0 else self.page_size
        #             buffer_indices[buffer_pos:buffer_pos + size] = torch.arange(
        #                 page * self.page_size, 
        #                 page * self.page_size + size
        #             )
        #             buffer_pos += size
        #         else:
        #             buffer_indices[buffer_pos:buffer_pos + self.page_size] = torch.arange(
        #                 page * self.page_size, 
        #                 (page + 1) * self.page_size
        #             )
        #             buffer_pos += self.page_size
            
        #     start_pos += offset

        # # 第二步：一次性提取数据并写入 buffer
        # self.compute_buffer[buffer_indices] = recv_tensor[indices]

    def send_kvcache_data(self, task_info):
        dst_rank = 2*task_info.cache_worker + 3
        token_num = task_info.token_num
        id = task_info.id
        # if id not in self.buffer_control_dict:
        #     print(f"[Worker][RANK {self.rank}] Error: id {id} not in buffer control dict")
        start_pos,offest = self._manage_buffer(id, token_num)
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 长度={token_num}")
        dist.send(tensor=self.compute_buffer[start_pos:offest], dst=dst_rank)
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 长度={token_num}")

    def send_kvcache_data_batch(self, combined_task_info):
        dst_rank = 2*combined_task_info.cache_worker + KVCACHE_offset
        token_num = combined_task_info.token_num
        id_pair_list = combined_task_info.id_token_pair
        # 计算总 token 数并预分配索引
        total_token_num = sum(id_pair.token_num for id_pair in id_pair_list)
        buffer_indices = torch.empty(total_token_num, dtype=torch.long)
        send_indices = torch.empty(total_token_num, dtype=torch.long)

        # 第一步：收集索引
        buffer_pos = 0
        send_pos = 0
        for id_pair in id_pair_list:
            i = id_pair.id
            token_num = id_pair.token_num
            
            if i not in self.page_manager.get_loaded_lists():
                print(f"[Worker][RANK {self.rank}][{time.time()}] Error: id {i} not in page manager")
            
            page_set = self.page_manager.access_item(i)
            
            # 生成索引
            for idx, page in enumerate(page_set):
                if idx == len(page_set) - 1:
                    size = token_num % self.page_size if token_num % self.page_size != 0 else self.page_size
                    buffer_indices[buffer_pos:buffer_pos + size] = torch.arange(
                        page * self.page_size,
                        page * self.page_size + size
                    )
                    send_indices[send_pos:send_pos + size] = torch.arange(
                        send_pos,
                        send_pos + size
                    )
                    buffer_pos += size
                    send_pos += size
                else:
                    buffer_indices[buffer_pos:buffer_pos + self.page_size] = torch.arange(
                        page * self.page_size,
                        (page + 1) * self.page_size
                    )
                    send_indices[send_pos:send_pos + self.page_size] = torch.arange(
                        send_pos,
                        send_pos + self.page_size
                    )
                    buffer_pos += self.page_size
                    send_pos += self.page_size

        # 第二步：一次性创建并填充 send_tensor
        send_tensor = torch.empty(
            (total_token_num,) + token_shape,
            dtype=torch.float16
        )
        send_tensor[send_indices] = self.compute_buffer[buffer_indices]

        # 第三步：移除保护
        for id_pair in id_pair_list:
            self.page_manager.remove_protected(id_pair.id)
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 长度={token_num}")
        dist.send(tensor=send_tensor, dst=dst_rank)
        if DEBUG:
            print(f"[Worker][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 长度={token_num}")
            

    def write_compute_buffer(self, task_info:Dict):
        cache_worker = task_info['cache_worker']
        token_num = task_info['token_num']
        # req_id = task_info['request_id']
        if task_info.get('id_token_pair') is not None:
            for i in task_info['id_token_pair']:
                self.buffer_control_dict[i] = []
        # ind = task_info['index']
        # start_pos,offest = self.buffer_control_dict[req_id][ind] 
        # NOTE: disable buffer management temporarily
        src_rank = 2*cache_worker + 3
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Writting kvcache data from Rank {src_rank}")
        recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
        )
        dist.recv(tensor=recv_tensor, src=src_rank)
        if DEBUG:
            print(f"[Worker][RANK {self.rank}] Recv tensor from Rank {src_rank}: {recv_tensor}")
        # 在这里使用to(device)会导致卡死
        # self.compute_buffer[start_pos:offest] = recv_tensor

    def preprare_send_data(self,task_info_list):
        # 这里的task_info同样有着一样的req_id
        coordinator_owner = self.coordinator_rank.owner()
        request_id = task_info_list[0]['request_id']
        cache_miss_dict = self.cache_miss_dict.get(request_id,{})
        send_task_list = []
        for task_info in task_info_list:
            item_id = task_info['id']
            # 这里能保证item_id在cache_miss_dict中吗？
            if cache_miss_dict.get(item_id) == CACHE_MISS:
                # print(f"[Worker][RANK {self.rank}] Cache miss detected")
                task_info['task_type'] = SIGNAL_RECV
                send_task_list.append(task_info)
            else:
                # 为什么会提前解除保护？
                if item_id in self.page_manager.get_loaded_lists():
                    self.page_manager.remove_protected(item_id)
        hit_rate = sum(cache_miss_dict.values())/len(cache_miss_dict)
        if hit_rate > 0.7 and DEBUG:
            print(f"[Worker][RANK {self.rank}] Request {request_id} Hit rate: {hit_rate} Sending {len(send_task_list)} tasks to kvcache") 
        # print(f"[Worker][RANK {self.rank}] Sending data to kvcache")
        if len(send_task_list)>0:
            rpc.rpc_sync(to=coordinator_owner, 
                            func=call_remote_method, 
                            args=(CacheCoordinator.add_requests,self.coordinator_rank, 
                                send_task_list))
        # 发buffer上的数据可能会被写掉？加锁？ 保证worker上的buffer没有被覆盖

    def forward_with_computation_grpc(self, tasks):
        if DEBUG:
            print(f"{now_time()}[Worker {self.rank}] Add {len(tasks)} requests to coordinator")
        tasks = TaskInfo_pb2.TaskInfoList(tasks=tasks)
        with grpc.insecure_channel(self.cache_coordinator_address) as channel:
            stub = TaskInfo_pb2_grpc.CacheCoordinatorServiceStub(channel)
            stub.ReceiveTasksFromInferWorker(tasks)  # 直接转发整个 TaskInfoList
        if DEBUG:
            print(f"{now_time()}[Worker {self.rank}] try to poll_batch")
        with grpc.insecure_channel(self.cache_coordinator_address) as channel:
            stub = TaskInfo_pb2_grpc.CacheCoordinatorServiceStub(channel)
            # stub.PollBatchFromInferWorker(tasks)
            time1 = time.time()
            future_call_poll = stub.PollBatchFromInferWorker.future(tasks)  # 直接转发整个 TaskInfoList
            if DEBUG:
                print(f"[Worker.forward_with_computation][RANK {self.rank}] finsh CacheCoordinator.poll_batch")
            cache_miss_dict_data = future_call_poll.result()
        time2 = time.time()
        print(f"[Worker.forward_with_computation][RANK {self.rank}] poll {len(tasks.tasks)} time {time2-time1}s")
        cache_miss_dict = json.loads(cache_miss_dict_data.msg)
        if DEBUG:
            print(f"[Worker.forward_with_computation][RANK {self.rank}] cache_miss_dict: {cache_miss_dict}")
        # cache_miss_dict = future_call_poll.result()
        for req_id in cache_miss_dict:
            self.cache_miss_dict[req_id] = cache_miss_dict[req_id]

    def preprare_send_data_grpc(self, task_info_list):
        send_task_list = []
        hit_counter = 0
        length_counter = 0
        for task_info in task_info_list:
            item_id = task_info.id
            request_id = task_info.request_id
            cache_miss_dict = self.cache_miss_dict.get(str(request_id),{})
            hit_counter += sum(cache_miss_dict.values())
            length_counter += len(cache_miss_dict)
            if cache_miss_dict.get(str(item_id)) == CACHE_MISS:
                # print(f"[Worker][RANK {self.rank}] Cache miss detected")
                task_info.task_type = SIGNAL_RECV
                send_task_list.append(task_info)
            else:
                if item_id in self.page_manager.get_loaded_lists():
                    self.page_manager.remove_protected(item_id)
        # TODO 适配
        hit_rate = hit_counter/length_counter
        # if hit_rate > 0.7 and DEBUG:
        print(f"[Worker {self.rank}]Hit rate: {hit_rate} Sending {len(send_task_list)} tasks to kvcache") 
        # print(f"[Worker][RANK {self.rank}] Sending data to kvcache")
        if len(send_task_list)>0:
            send_task_list_gprc = TaskInfo_pb2.TaskInfoList(tasks=send_task_list)
            with grpc.insecure_channel(self.cache_coordinator_address) as channel:
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