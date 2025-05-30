from functools import cache
import json
from typing import Dict, Tuple
from datetime import datetime
import random

from protos import TaskInfo_pb2, TaskInfo_pb2_grpc

import grpc
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from DistributedStorage.Signals import SIGNAL_SEND, SIGNAL_RECV
from rpc_def import KVCACHE_offset,WORKER_offset, WORKER_NUM, KVCACHE_NUM
from Utils.channelpool import ChannelPool
from Remote.remote_call import call_remote_method
from Model.qwen2 import token_shape
from config import *
import time
import os
import logging
SM_DEBUG=1

# from rdma_transport import RDMAEndpoint
from rdma_onesided_transport import RDMAOneSidedEndpoint
import ipc_service

torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

class KVCache(TaskInfo_pb2_grpc.KVCacheServiceServicer):
    def __init__(self, rank, cache_size, page_size, master_port, rank_to_ip, rank_to_ip_rdma, server):
        self.rank = rank
        self.cache_index = int(rank/2) -1
        self.gpu_index = 0
        self.cache_size = cache_size
        self.device = torch.device(f"cuda:{0}")
        self.cpu_cache_data = torch.full(
            (self.cache_size,) + token_shape, 
            self.rank,
            device='cpu',
            dtype=torch.float16
        )
        self.start_pos = 0
        self.page_size = page_size
        self.recv_counter = 0
        self.send_counter = 0
        self.master_port = master_port
        self.server = server
        self.rank_to_ip_grpc = rank_to_ip
        self.rank_to_ip_rdma = rank_to_ip_rdma

        # for shared memory
        # self.cuda_cache_data= torch.full(
        #     (self.cache_size,) + token_shape, 
        #     self.rank,
        #     device=self.device,
        #     dtype=torch.float16
        # )
        self.max_pages=(self.cache_size+page_size-1)//page_size 
        self.src_index_pool = torch.empty(self.max_pages, dtype=torch.int64, device=self.device)
        self.dst_index_pool = torch.empty(self.max_pages, dtype=torch.int64, device=self.device)
        self.src_index_pool.fill_(-1)  # 用无效值初始化
        self.dst_index_pool.fill_(-1)
        self.shm_name = f"/sj_kv_cache_{self.cache_index}"
        # try:
        #     # 先尝试清理可能存在的残留共享内存
        #     ipc_service.producer_cleanup()  
        # except:
        #     pass
        # try:
        #     ipc_service.producer_init(self.cache_index, self.shm_name.encode())
        # except Exception as e:
        #     print(f"KVcache {self.rank} shared mem init failed: {e}")
        # self.cpu_cache_data = self.cpu_cache_data.pin_memory()  # 后pin内存
        # torch.cuda.synchronize(self.device)
        
        self.channelpool = ChannelPool()
        print(f"[KVcach][RANK {self.rank}] Init finish")
        
    def get_index_tensors(self, num_pages):
        assert num_pages <= self.max_pages, f"Requested {num_pages} exceeds pool size {self.max_pages}"
        return (
            self.src_index_pool[:num_pages],  # 返回视图，零拷贝
            self.src_index_pool[:num_pages]
        )

    def start_rdma(self):
        self.ep = RDMAOneSidedEndpoint(self.rank_to_ip_rdma[self.rank], str(self.master_port+10), "server")
        self.buffer_size = 1024*1024*1024 
        if self.ep.run_server(max_clients=WORKER_NUM, local_cpu_size=self.buffer_size, local_gpu_size=0, hugepage=False) != 0:
            logging.info("Server failed to accept clients.")
            assert (0)
        logging.info(f"RDMA server started at {self.rank_to_ip_rdma[self.rank]}:{str(self.master_port+10)}")
        
    def send_data_batch(self,combined_task_info:Dict):
        dst_rank = 2*combined_task_info['infer_worker'] + WORKER_offset
        token_num = combined_task_info['token_num']
        id_token_pair_list = combined_task_info['id_token_pair']
        cache_pages_list = combined_task_info['cache_pages_list']
        # 计算总 token 数
        total_token_num = sum(id_token_pair[1] for id_token_pair in id_token_pair_list)

        # 一次性分配大 tensor
        send_tensor = torch.empty(
            (total_token_num,) + token_shape,
            dtype=torch.float16
        )
        # time0 = time.time()
        # self.ep.post_rdma_write(rank=dst_rank, size=token_num*128*8*28, src_type="cpu", dst_type="cpu")
        # dist.send(tensor=send_tensor, dst=dst_rank)
        # time1 = time.time()
        # logging.info(f"{self.rank} send once time: {time1-time0}s, throughput: {(total_token_num*128*28*8/(time1-time0)/(1e9))} GB/s")
   

    def receive_data_batch(self, combined_task_info:Dict):
        # request_id = task_info['request_id']
        infer_worker = combined_task_info['infer_worker']
        token_num = combined_task_info['token_num']
        id_token_pair_list = combined_task_info['id_token_pair']
        cache_pages_list = combined_task_info['cache_pages_list']
        src_rank = 2*infer_worker + WORKER_offset
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 开始接收数据从 Rank {src_rank} 长度为{token_num}")
        recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
        )
        # dist.recv(tensor=recv_tensor, src=src_rank)
        # self.ep.post_rdma_read(rank=src_rank, size=token_num*128*8*28, src_type="cpu", dst_type="cpu")
    
    def shared_data_batch_pages(self, combined_task_info: Dict):
        """使用CUDA共享内存传输数据（带详细性能分析）"""
        total_time_start = time.perf_counter()
        dst_rank = 2*combined_task_info['infer_worker'] + WORKER_offset
        token_num = combined_task_info['token_num']
        id_token_pair_list = combined_task_info['id_token_pair']
        cache_pages_list = combined_task_info['cache_pages_list']
        
        if SM_DEBUG:
            print(f"\n[KVCache][Rank {self.rank}] 开始共享数据到 Rank {dst_rank}, token数={token_num}")

        # ==================== 1. 数据准备阶段 ====================
        prep_start = time.perf_counter()
        
        # 计算总token数
        total_token_num = sum(id_token_pair[1] for id_token_pair in id_token_pair_list)
        
        # 准备源数据（从CPU到GPU）
        data_transfer_start = time.perf_counter()
        src_data = self.cuda_cache_data
        data_transfer_end = time.perf_counter()
        
        prep_end = time.perf_counter()
        if SM_DEBUG:
            print(f"[阶段1] 数据准备耗时: {(prep_end-prep_start)*1000:.2f}ms (含GPU传输: {(data_transfer_end-data_transfer_start)*1000:.2f}ms)")

        # ==================== 2. 页面索引计算 ====================
        index_start = time.perf_counter()
        
        page_indices = []
        page_offsets = []
        current_offset = 0
        
        for idx, page_list in enumerate(cache_pages_list):
            item_token_num = id_token_pair_list[idx][1]
            for page_idx, page in enumerate(page_list):
                # 计算每个页面的实际大小
                size = (item_token_num % self.page_size) if (page_idx == len(page_list)-1 and 
                        item_token_num % self.page_size != 0) else self.page_size
                
                page_indices.append(page)
                page_offsets.append(current_offset)
                current_offset += size
        
        index_end = time.perf_counter()
        if SM_DEBUG:
            print(f"[阶段2] 页面索引计算耗时: {(index_end-index_start)*1000:.2f}ms")
            print(f"        总页面数: {len(page_indices)}, 总元素数: {current_offset}")

        # ==================== 3. 张量转换 ====================
        tensor_start = time.perf_counter()
        # 获取预分配的张量
        num_pages = len(combined_task_info['cache_pages_list'])
        src_offsets, dest_offsets = self.get_index_tensors(num_pages)
        # 转换为CUDA张量
        src_offsets = torch.tensor(
            [p * self.page_size for p in page_indices],
            dtype=torch.int64, 
            device=self.device
        )
        dest_offsets = torch.tensor(
            page_offsets,
            dtype=torch.int64, 
            device=self.device
        )
         # 创建TensorDims对象而不是字典
        dims = ipc_service.TensorDims()
        dims.total_tokens = len(page_indices) * self.page_size
        dims.head_size = 128
        dims.num_kv_heads = 2
        dims.num_layers = 28
        dims.kv_pair = 2
        # 计算总数据量
        total_tokens = len(page_indices) * self.page_size
        data_size = total_tokens*128*8*28
        tensor_end = time.perf_counter()
        if SM_DEBUG:
            print(f"[阶段3] 张量转换耗时: {(tensor_end-tensor_start)*1000:.2f}ms")

        # ==================== 4. IPC页面拷贝 ====================
        # dims = [
        # self.page_size *src_offsets.shape[0] ,  # 会被覆盖为总token数
        # 128,2,28,2
        # ]
         # 创建TensorDims对象而不是字典
        ipc_start = time.perf_counter()

        ipc_service.producer_copy_pages(
            src_data,
            src_offsets,
            dest_offsets,
            self.page_size,
            dims
        )
        
        ipc_end = time.perf_counter()
        if SM_DEBUG:
            print(f"[阶段4] IPC页面拷贝耗时: {(ipc_end-ipc_start)*1000:.2f}ms")
            print(f"total_tokens:{total_tokens}  数据量: {data_size/1024/1024:.2f}MB, " 
                f"带宽: {data_size/(ipc_end-ipc_start)/1024/1024:.2f}MB/s")

        # ==================== 5. 总体统计 ====================
        total_time_end = time.perf_counter()
        total_time = total_time_end - total_time_start
        if SM_DEBUG:
            print(f"[阶段5] 总耗时: {total_time*1000:.2f}ms")
            print(f"        平均每token耗时: {total_time/token_num*1000:.2f}ms")
            print(f"[KVCache][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}")
    
    def shared_data_batch_pages_cpu2gpu(self, combined_task_info: Dict):
        """使用CUDA共享内存传输数据（带详细性能分析）"""
        total_time_start = time.perf_counter()
        dst_rank = 2*combined_task_info['infer_worker'] + WORKER_offset
        token_num = combined_task_info['token_num']
        id_token_pair_list = combined_task_info['id_token_pair']
        cache_pages_list = combined_task_info['cache_pages_list']
        
        if SM_DEBUG:
            print(f"\n[KVCache][Rank {self.rank}] 开始共享数据到 Rank {dst_rank}, token数={token_num}")

        # ==================== 1. 数据准备阶段 ====================
        prep_start = time.perf_counter()
        
        # 计算总token数
        total_token_num = sum(id_token_pair[1] for id_token_pair in id_token_pair_list)
        
        # 准备源数据（从CPU到GPU）
        data_transfer_start = time.perf_counter()
        src_data = self.cpu_cache_data
        data_transfer_end = time.perf_counter()
        
        prep_end = time.perf_counter()
        if SM_DEBUG:
            print(f"[阶段1] 数据准备耗时: {(prep_end-prep_start)*1000:.2f}ms (含GPU传输: {(data_transfer_end-data_transfer_start)*1000:.2f}ms)")

        # ==================== 2. 页面索引计算 ====================
        index_start = time.perf_counter()
        
        page_indices = []
        page_offsets = []
        current_offset = 0
        
        for idx, page_list in enumerate(cache_pages_list):
            item_token_num = id_token_pair_list[idx][1]
            for page_idx, page in enumerate(page_list):
                # 计算每个页面的实际大小
                size = (item_token_num % self.page_size) if (page_idx == len(page_list)-1 and 
                        item_token_num % self.page_size != 0) else self.page_size
                
                page_indices.append(page)
                page_offsets.append(current_offset)
                current_offset += size
        
        index_end = time.perf_counter()
        if SM_DEBUG:
            print(f"[阶段2] 页面索引计算耗时: {(index_end-index_start)*1000:.2f}ms")
            print(f"        总页面数: {len(page_indices)}, 总元素数: {current_offset}")

        # ==================== 3. 张量转换 ====================
        tensor_start = time.perf_counter()
        # 获取预分配的张量
        num_pages = len(combined_task_info['cache_pages_list'])
        src_offsets, dest_offsets = self.get_index_tensors(num_pages)
        # # 转换为CUDA张量
        # src_offsets = torch.tensor(
        #     [p * self.page_size for p in page_indices],
        #     dtype=torch.int64, 
        #     device=self.device
        # )
        # dest_offsets = torch.tensor(
        #     page_offsets,
        #     dtype=torch.int64, 
        #     device=self.device
        # )
        # 在张量创建前强制同步CUDA设备
        torch.cuda.synchronize(device=self.device)  # 确保之前的所有CUDA操作完成

        try:
            src_offsets = torch.tensor(
                [p * self.page_size for p in page_indices],
                dtype=torch.int64,
                device=self.device
            )
            dest_offsets = torch.tensor(
            page_offsets,
            dtype=torch.int64, 
            device=self.device
            )
        except RuntimeError as e:
            print(f"CUDA error during tensor creation: {e}")
            torch.cuda.empty_cache()  # 紧急清理缓存
            raise
         # 创建TensorDims对象而不是字典
        dims = ipc_service.TensorDims()
        dims.total_tokens = len(page_indices) * self.page_size
        dims.head_size = 128
        dims.num_kv_heads = 2
        dims.num_layers = 28
        dims.kv_pair = 2
        # 计算总数据量
        total_tokens = len(page_indices) * self.page_size
        data_size = total_tokens*128*8*28
        tensor_end = time.perf_counter()
        if SM_DEBUG:
            print(f"[阶段3] 张量转换耗时: {(tensor_end-tensor_start)*1000:.2f}ms")

        # ==================== 4. IPC页面拷贝 ====================
        # dims = [
        # self.page_size *src_offsets.shape[0] ,  # 会被覆盖为总token数
        # 128,2,28,2
        # ]
         # 创建TensorDims对象而不是字典
        ipc_start = time.perf_counter()

        ipc_service.producer_zero_copy_pages(
            src_data,
            src_offsets,
            dest_offsets,
            self.page_size,
            dims
        )
        
        ipc_end = time.perf_counter()
        if SM_DEBUG:
            print(f"[阶段4] IPC页面拷贝耗时: {(ipc_end-ipc_start)*1000:.2f}ms")
            print(f"total_tokens:{total_tokens}  数据量: {data_size/1024/1024:.2f}MB, " 
                f"带宽: {data_size/(ipc_end-ipc_start)/1024/1024:.2f}MB/s")

        # ==================== 5. 总体统计 ====================
        total_time_end = time.perf_counter()
        total_time = total_time_end - total_time_start
        if SM_DEBUG:
            print(f"[阶段5] 总耗时: {total_time*1000:.2f}ms")
            print(f"        平均每token耗时: {total_time/token_num*1000:.2f}ms")
            print(f"[KVCache][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}")

    def receive_task_info_batch_gprc(self, task_info_list): ## only support send from kvcache to worker, all tasks have the same cache worker        
        combined_task_info = {} ## key: infer worker
        confirmation_msg = {} ## key: request id
        if DEBUG:
            print(f"[KVCache {self.rank}] receive_task_info_batch_gprc len:{len(task_info_list)}")
        for task_info in task_info_list:
            infer_worker = task_info.infer_worker
            if infer_worker !=0 and DEBUG:
                print(f"[KVCache][RANK {self.rank}] infer worker is {infer_worker}") 
            cache_worker = task_info.cache_worker
            req_id = task_info.request_id
            token_num = task_info.token_num
            item_id = task_info.id
            cache_pages_list = task_info.cache_pages_list
            if item_id == -1:
                continue
            # 到底是什么时候需要管理缓存？
            # 为什么只在RECV管理时会出现key error？
            if combined_task_info.get(infer_worker) == None:
                combined_task_info[infer_worker] = {}
            if task_info.task_type == SIGNAL_SEND:
                if combined_task_info[infer_worker].get(SIGNAL_SEND) == None:
                    combined_task_info[infer_worker][SIGNAL_SEND] = {"infer_worker":infer_worker, 
                                                                "cache_worker":cache_worker,
                                                                "token_num":token_num,
                                                                'task_type': SIGNAL_SEND,
                                                                'id_token_pair':[(item_id,token_num)],
                                                                'cache_pages_list':[cache_pages_list],
                                                                } 
                else:
                    combined_task_info[infer_worker][SIGNAL_SEND]['token_num'] += token_num
                    combined_task_info[infer_worker][SIGNAL_SEND]['id_token_pair'].append((item_id,token_num))
                    combined_task_info[infer_worker][SIGNAL_SEND]['cache_pages_list'].append(cache_pages_list)
            if task_info.task_type == SIGNAL_RECV:
                if combined_task_info[infer_worker].get(SIGNAL_RECV) == None:
                    combined_task_info[infer_worker][SIGNAL_RECV] = {"infer_worker":infer_worker, 
                                                                "cache_worker":cache_worker,
                                                                "token_num":token_num,
                                                                'task_type': SIGNAL_RECV,
                                                                'id_token_pair':[(item_id,token_num)],
                                                                'cache_pages_list':[cache_pages_list],
                                                                }    
                else:
                    combined_task_info[infer_worker][SIGNAL_RECV]['token_num'] += token_num
                    combined_task_info[infer_worker][SIGNAL_RECV]['id_token_pair'].append((item_id,token_num))
                    combined_task_info[infer_worker][SIGNAL_RECV]['cache_pages_list'].append(cache_pages_list)
            if confirmation_msg.get(req_id) == None:
                confirmation_msg[req_id] = 1
            else:
                confirmation_msg[req_id] += 1
        # 初始状态下，第一轮是全空的任务
        for task_infer_worker in combined_task_info:
            combined_task_list = combined_task_info[task_infer_worker]
            for task_info in combined_task_list.values():
                task_type = task_info['task_type']
                infer_worker = task_info['infer_worker']
                infer_worker_port = self.master_port + 2*infer_worker + WORKER_offset
                infer_worker_addr = f"{self.rank_to_ip_grpc[2*infer_worker + WORKER_offset]}:{infer_worker_port}"
                if task_type == SIGNAL_SEND:
                    if DEBUG:
                        print(f"[KVCache.receive_task_info_batch][RANK {self.rank}]{task_info}")
                        print(f"[KVCache {self.rank}] 执行Send请求 - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
                    combined_task_info_pb = self._task_info_json_to_pb(task_info)
                    # if cache_worker== infer_worker:
                    #     self.shared_data_batch_pages(task_info)
                    # with grpc.insecure_channel(infer_worker_addr) as channel:
                    channel = self.channelpool.get_channel(infer_worker_addr)
                    stub = TaskInfo_pb2_grpc.InferWorkerServiceStub(channel)
                    remote_recv = stub.RecvKVCacheData.future(combined_task_info_pb)
                    if cache_worker == infer_worker:
                        remote_recv.result()
                    else:
                        self.send_data_batch(task_info)
                        remote_recv.result()
                    self.send_counter += 1

                elif task_type == SIGNAL_RECV:
                    cache_worker = task_info['cache_worker']
                    if DEBUG:
                        print(f"[KVCache.receive_task_info_batch][RANK {self.rank}] 执行Recv请求 - workerRank {2*infer_worker+2} -> cacheRank {2*cache_worker+3}")
                        print(f"[KVCache {self.rank}] 执行Recv请求 - workerRank {2*infer_worker+2} -> cacheRank {2*cache_worker+3}")
                    combined_task_info_pb = self._task_info_json_to_pb(task_info)
                    # with grpc.insecure_channel(infer_worker_addr) as channel:
                    channel = self.channelpool.get_channel(infer_worker_addr)
                    stub = TaskInfo_pb2_grpc.InferWorkerServiceStub(channel)
                    remote_send = stub.SendKVCacheData.future(combined_task_info_pb)
                    self.receive_data_batch(task_info)
                    remote_send.result()
                    now = datetime.now()
                    nowtime = now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"
                    self.recv_counter += 1
        return confirmation_msg


    def ReceiveTasksFromCoordinator(self, request, context):
        confirmation_msg = self.receive_task_info_batch_gprc(request.tasks)
        comfirmation_data = json.dumps(confirmation_msg)
        return TaskInfo_pb2.ComfirmationMessage(msg = comfirmation_data)
    
    def _task_info_json_to_pb(self, task_info:Dict):
        combined_task_info_pb = TaskInfo_pb2.CombindedTaskInfo()
        combined_task_info_pb.infer_worker = task_info['infer_worker']
        combined_task_info_pb.cache_worker = task_info['cache_worker']
        combined_task_info_pb.token_num = task_info['token_num']
        combined_task_info_pb.task_type = task_info['task_type']
        combined_task_info_pb.cache_pages_list.extend([TaskInfo_pb2.PageList(cache_pages_list=page_list) for page_list in task_info['cache_pages_list']])
        combined_task_info_pb.id_token_pair.extend([TaskInfo_pb2.IdTokenPair(id=id_token_pair[0], token_num=id_token_pair[1]) for id_token_pair in task_info['id_token_pair']])
        return combined_task_info_pb
    
    def ShutDown(self, request, context):
        self.server.stop(0)
        return TaskInfo_pb2.Empty()
    
    def send_confirmation(self, confirmation_msg):
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 发送确认消息到调度器: 请求ID={confirmation_msg}")
        return confirmation_msg

    def terminate(self):
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 收到终止信号，退出运行")
        self.show_counter()
        return "Terminated"

    def show_counter(self):
        print(f"[KVCache][RANK {self.rank}] send_counter: {self.send_counter}, recv_counter: {self.recv_counter}")
