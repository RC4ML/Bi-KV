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
from Remote.remote_call import call_remote_method
from Model.qwen2 import token_shape
from config import *
import time
import os
import logging

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
        self.cache_data = torch.full(
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
        # self.rank_to_ip_rdma =  {0:'10.0.0.2',
        #                         1:'10.0.0.2',
        #                         2:'10.0.0.3',
        #                         3:'10.0.0.3',
        #                         4:'10.0.0.4',
        #                         5:'10.0.0.4'
        #                         }#
        # self.rank_to_ip_rdma =  {0:'10.0.0.2',
        #                         1:'10.0.0.2',
        #                         2:'10.0.0.2',
        #                         3:'10.0.0.2',
        #                         4:'10.0.0.1',
        #                         5:'10.0.0.1',
        #                         6:'10.0.0.3',
        #                         7:'10.0.0.3',
        #                         8:'10.0.0.4',
        #                         9:'10.0.0.4'
        #                         }#
        # for shared memory
        self.shm_name = f"/kv_cache_{self.cache_index}"  # 唯一共享内存名称
        self._init_shared_memory()

    def _init_shared_memory(self):
        """初始化CUDA共享内存区域"""
        device_id = self.gpu_index 
        try:
            # 先尝试清理可能存在的残留共享内存
            ipc_service.producer_cleanup()  
        except:
            pass
        buffer_size = self.cache_data.element_size() * self.cache_data.nelement()
        print(f"buffer_size:{buffer_size/(1024**2)}MB")
        # 初始化生产者端共享内存
        ipc_service.producer_init(device_id, self.shm_name.encode(), buffer_size)

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
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 长度={token_num}")
        # 计算总 token 数
        total_token_num = sum(id_token_pair[1] for id_token_pair in id_token_pair_list)

        # 一次性分配大 tensor
        send_tensor = torch.empty(
            (total_token_num,) + token_shape,
            dtype=torch.float16
        )

        # indices = torch.empty(total_token_num, dtype=torch.long)
        # offset = 0
        # circle_counter = 0
        # for idx, page_list in enumerate(cache_pages_list):
        #     id_token_pair = id_token_pair_list[idx]
        #     item_token_num = id_token_pair[1]
        #     for page_idx, page in enumerate(page_list):
        #         start = page * self.page_size
        #         circle_counter += 1
        #         if page_idx == len(page_list) - 1:
        #             size = (item_token_num % self.page_size) if (item_token_num % self.page_size != 0) else self.page_size
        #             indices[offset:offset + size] = torch.arange(start, start + size)
        #             offset += size
        #         else:
        #             indices[offset:offset + self.page_size] = torch.arange(start, start + self.page_size)
        #             offset += self.page_size

        # send_tensor[:] = self.cache_data[indices]
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] send_tensor shape: {send_tensor.size()} token num: {token_num}")
        # time0 = time.time()
        self.ep.post_rdma_write(rank=dst_rank, size=token_num*128*8*28, src_type="cpu", dst_type="cpu")
        # dist.send(tensor=send_tensor, dst=dst_rank)
        # time1 = time.time()
        # logging.info(f"{self.rank} send once time: {time1-time0}s, throughput: {(total_token_num*128*28*8/(time1-time0)/(1e9))} GB/s")
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 长度={token_num}")

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
        self.ep.post_rdma_read(rank=src_rank, size=token_num*128*8*28, src_type="cpu", dst_type="cpu")
        # if DEBUG:
        #     print(f"[KVCache][CPU {self.cache_index}] [rank{self.rank}] 完成接收数据从 Rank {infer_worker} [rank{src_rank}]")
        # # 计算总 token 数
        # total_token_num = sum(id_token_pair[1] for id_token_pair in id_token_pair_list)

        # # 预分配索引 tensor
        # recv_indices = torch.empty(total_token_num, dtype=torch.long)
        # cache_indices = torch.empty(total_token_num, dtype=torch.long)

        # # 第一步：收集索引
        # start_pos = 0
        # cache_pos = 0
        # for idx, pages_list in enumerate(cache_pages_list):
        #     id_token_pair = id_token_pair_list[idx]
        #     item_token_num = id_token_pair[1]
            
        #     # 生成 recv_tensor 的索引
        #     recv_indices[start_pos:start_pos + item_token_num] = torch.arange(
        #         start_pos, 
        #         start_pos + item_token_num
        #     )
            
        #     # 生成 cache_data 的索引
        #     for page_idx, page in enumerate(pages_list):
        #         if page_idx == len(pages_list) - 1:
        #             size = (item_token_num % self.page_size) if (item_token_num % self.page_size != 0) else self.page_size
        #             cache_indices[cache_pos:cache_pos + size] = torch.arange(
        #                 page * self.page_size, 
        #                 page * self.page_size + size
        #             )
        #             cache_pos += size
        #         else:
        #             cache_indices[cache_pos:cache_pos + self.page_size] = torch.arange(
        #                 page * self.page_size, 
        #                 (page + 1) * self.page_size
        #             )
        #             cache_pos += self.page_size
            
        #     start_pos += item_token_num

        # # 第二步：一次性写入 cache
        # self.cache_data[cache_indices] = recv_tensor[recv_indices]

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
                    # run_grpc_client(self.rdma_client, '192.168.189.9', 50052, 0, 1024*1024*128)
                    # if cache_worker== infer_worker:
                    #     self.shared_data_batch(task_info)
                    with grpc.insecure_channel(infer_worker_addr) as channel:
                        stub = TaskInfo_pb2_grpc.InferWorkerServiceStub(channel)
                        remote_recv = stub.RecvKVCacheData.future(combined_task_info_pb)
                        if cache_worker == infer_worker:
                            self.shared_data_batch(task_info)
                        else:
                            self.send_data_batch(task_info)
                        remote_recv.result()
                    # print(f"[KVCache][RANK {self.rank}] 执行Send请求完成 - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
                    self.send_counter += 1

                elif task_type == SIGNAL_RECV:
                    cache_worker = task_info['cache_worker']
                    if DEBUG:
                        print(f"[KVCache.receive_task_info_batch][RANK {self.rank}] 执行Recv请求 - workerRank {2*infer_worker+2} -> cacheRank {2*cache_worker+3}")
                        print(f"[KVCache {self.rank}] 执行Recv请求 - workerRank {2*infer_worker+2} -> cacheRank {2*cache_worker+3}")
                    combined_task_info_pb = self._task_info_json_to_pb(task_info)
                    with grpc.insecure_channel(infer_worker_addr) as channel:
                        stub = TaskInfo_pb2_grpc.InferWorkerServiceStub(channel)
                        remote_send = stub.SendKVCacheData.future(combined_task_info_pb)
                        self.receive_data_batch(task_info)
                        remote_send.result()
                    now = datetime.now()
                    nowtime = now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"
                    # print(f"[KVCache][RANK {self.rank}] 执行Recv请求完成 - workerRank {2*infer_worker+2} -> cacheRank {2*cache_worker+3}")
                    self.recv_counter += 1
        return confirmation_msg


    def ReceiveTasksFromCoordinator(self, request, context):
        # print(f"[KVCache {self.rank}]收到Coordinator请求 长度为{len(request.tasks)}")
        confirmation_msg = self.receive_task_info_batch_gprc(request.tasks)
        # 纯控制测试用
        # confirmation_msg = {}
        # for task_info in request.tasks:
        #     req_id = task_info.request_id
        #     if task_info.id==-1:
        #         continue
        #     if confirmation_msg.get(req_id) == None:
        #         confirmation_msg[req_id] = 1
        #     else:
        #         confirmation_msg[req_id] += 1
        # confirmation_msg是dict，需要转成字符串后传输
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

    def shared_data_batch(self, combined_task_info: Dict):
        """使用CUDA共享内存传输数据"""
        dst_rank = 2*combined_task_info['infer_worker'] + WORKER_offset
        token_num = combined_task_info['token_num']
        id_token_pair_list = combined_task_info['id_token_pair']
        cache_pages_list = combined_task_info['cache_pages_list']
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 长度={token_num}")
        # 计算总 token 数
        total_token_num = sum(id_token_pair[1] for id_token_pair in id_token_pair_list)

        # 一次性分配大 tensor
        # send_tensor = torch.empty(
        #     (total_token_num,) + token_shape,
        #     dtype=torch.float16,
        #     device='cuda'
        # )
        send_tensor = torch.ones(
            (total_token_num,) + token_shape,
            dtype=torch.float16,
            device='cuda'
        )
         # 添加形状校验
        # expected_numel = total_token_num * np.prod(token_shape)
        # assert send_tensor.numel() == expected_numel, \
        #     f"形状不匹配: {send_tensor.shape} vs 预期{token_shape}"
        # indices = torch.empty(total_token_num, dtype=torch.long)
        # offset = 0
        # circle_counter = 0
        # for idx, page_list in enumerate(cache_pages_list):
        #     id_token_pair = id_token_pair_list[idx]
        #     item_token_num = id_token_pair[1]
        #     for page_idx, page in enumerate(page_list):
        #         start = page * self.page_size
        #         circle_counter += 1
        #         if page_idx == len(page_list) - 1:
        #             size = (item_token_num % self.page_size) if (item_token_num % self.page_size != 0) else self.page_size
        #             indices[offset:offset + size] = torch.arange(start, start + size)
        #             offset += size
        #         else:
        #             indices[offset:offset + self.page_size] = torch.arange(start, start + self.page_size)
        #             offset += self.page_size

        # send_tensor[:] = self.cache_data[indices]
        if DEBUG:
            print(f"[KVCache]共享内存[Rank {self.rank}] send_tensor shape: {send_tensor.size()} token num: {token_num}")
        time0 = time.time()
        ipc_service.producer_send(send_tensor)
        time1 = time.time()
        if DEBUG:
            print(f"内部shared once time: {time1-time0}s, total_token_num: {total_token_num},throughput: {((token_num*8*28*128)/(time1-time0)/(1e9))} GB/s")
        if DEBUG:
            print(f"[KVCache]共享内存[Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 长度={token_num}")