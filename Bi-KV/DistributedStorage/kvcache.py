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
from rpc_def import KVCACHE_offset,WORKER_offset
from Remote.remote_call import call_remote_method
from Model.qwen2 import token_shape
from config import *
import time



class KVCache(TaskInfo_pb2_grpc.KVCacheServiceServicer):
    def __init__(self, rank, cache_size, page_size, master_port, server):
        self.rank = rank
        self.cache_index = int(rank/2) -1
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
        if DEBUG:
            print(f"[KVCache][CPU index:{self.cache_index} rank: {self.rank}] 初始化：Tensor大小={self.cache_data.size()}，值={self.rank}")

    def send_data(self,task_info:Dict):
        dst_rank = 2*task_info['infer_worker'] + 2
        request_id = task_info['request_id']
        token_num = task_info['token_num']
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={token_num}")
        start_pos = random.randint(0,self.cache_size/2)
        send_tensor = self.cache_data[start_pos:start_pos+token_num]
        dist.send(tensor=send_tensor, dst=dst_rank)
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={token_num}")

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

        indices = torch.empty(total_token_num, dtype=torch.long)
        offset = 0
        circle_counter = 0
        for idx, page_list in enumerate(cache_pages_list):
            id_token_pair = id_token_pair_list[idx]
            item_token_num = id_token_pair[1]
            for page_idx, page in enumerate(page_list):
                start = page * self.page_size
                circle_counter += 1
                if page_idx == len(page_list) - 1:
                    size = (item_token_num % self.page_size) if (item_token_num % self.page_size != 0) else self.page_size
                    indices[offset:offset + size] = torch.arange(start, start + size)
                    offset += size
                else:
                    indices[offset:offset + self.page_size] = torch.arange(start, start + self.page_size)
                    offset += self.page_size

        send_tensor[:] = self.cache_data[indices]
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] send_tensor shape: {send_tensor.size()} token num: {token_num}")
        time0 = time.time()
        dist.send(tensor=send_tensor, dst=dst_rank)
        time1 = time.time()
        # print(f"send once time: {time1-time0}s, throughput: {((token_num*8*28*128)/(time1-time0)/(1e9))} GB/s")
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 长度={token_num}")

    def receive_data(self, task_info:Dict):
        # request_id = task_info['request_id']
        infer_worker = task_info['infer_worker']
        token_num = task_info['token_num']
        item_id = task_info['id']
        src_rank = 2*infer_worker + 2
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 开始接收数据从 Rank {src_rank}")
        recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
        )
        dist.recv(tensor=recv_tensor, src=src_rank)
        if DEBUG:
            print(f"[KVCache][CPU {self.cache_index}] [rank{self.rank}] 完成接收数据从 Rank {infer_worker} [rank{src_rank}]")
        next_pos = self.start_pos + token_num
        if next_pos > self.cache_size:
            self.start_pos = 0
            next_pos = self.start_pos + token_num
        # self.cache_data[self.start_pos:next_pos] = recv_tensor
        self.start_pos = next_pos

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
        dist.recv(tensor=recv_tensor, src=src_rank)
        if DEBUG:
            print(f"[KVCache][CPU {self.cache_index}] [rank{self.rank}] 完成接收数据从 Rank {infer_worker} [rank{src_rank}]")
        # 计算总 token 数
        total_token_num = sum(id_token_pair[1] for id_token_pair in id_token_pair_list)

        # 预分配索引 tensor
        recv_indices = torch.empty(total_token_num, dtype=torch.long)
        cache_indices = torch.empty(total_token_num, dtype=torch.long)

        # 第一步：收集索引
        start_pos = 0
        cache_pos = 0
        for idx, pages_list in enumerate(cache_pages_list):
            id_token_pair = id_token_pair_list[idx]
            item_token_num = id_token_pair[1]
            
            # 生成 recv_tensor 的索引
            recv_indices[start_pos:start_pos + item_token_num] = torch.arange(
                start_pos, 
                start_pos + item_token_num
            )
            
            # 生成 cache_data 的索引
            for page_idx, page in enumerate(pages_list):
                if page_idx == len(pages_list) - 1:
                    size = (item_token_num % self.page_size) if (item_token_num % self.page_size != 0) else self.page_size
                    cache_indices[cache_pos:cache_pos + size] = torch.arange(
                        page * self.page_size, 
                        page * self.page_size + size
                    )
                    cache_pos += size
                else:
                    cache_indices[cache_pos:cache_pos + self.page_size] = torch.arange(
                        page * self.page_size, 
                        (page + 1) * self.page_size
                    )
                    cache_pos += self.page_size
            
            start_pos += item_token_num

        # 第二步：一次性写入 cache
        self.cache_data[cache_indices] = recv_tensor[recv_indices]

    def send_confirmation(self, confirmation_msg):
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 发送确认消息到调度器: 请求ID={confirmation_msg}")
        return confirmation_msg

    def terminate(self):
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 收到终止信号，退出运行")
        self.show_counter()
        return "Terminated"
    
    def receive_task_info(self, task_info:Dict, worker_ref):
        from Worker.Worker import Worker
        if DEBUG:
            print(f"[KVCache][RANK {self.rank}] taskinfo is {task_info}")
        # task_type, request_id, cache_worker, infer_worker = task_info
        infer_worker = task_info['infer_worker']
        cache_worker = task_info['cache_worker']
        task_type = task_info['task_type']
        request_id = task_info['request_id']
        # return request_id
        if task_type == SIGNAL_SEND:
            if DEBUG:
                print(f"[KVCache.receive_task_info][RANK {self.rank}]{task_info}")
                print(f"[KVCache.receive_task_info[RANK {self.rank}] 执行Send请求 - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
            remote_recv = rpc.rpc_async(
                to=worker_ref.owner(), func=call_remote_method, 
                args=(Worker.receive_kvcache_data_batch,worker_ref, task_info))
            self.send_data(task_info)
            remote_recv.wait()
            if DEBUG:
                print(f"[KVCache.receive_task_info][RANK {self.rank}] Send请求完成 - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
            return request_id
        elif task_type == SIGNAL_RECV:
            cache_worker = task_info['cache_worker']
            infer_worker = task_info['infer_worker']
            if DEBUG:
                print(f"[KVCache] 执行请求 {request_id} - Rank {infer_worker+WORKER_offset} -> Rank {cache_worker+KVCACHE_offset}")
            remote_send = rpc.rpc_async(
                to=worker_ref.owner(), func=call_remote_method, 
                args=(Worker.send_kvcache_data_batch,worker_ref, task_info))
            self.receive_data(task_info)
            remote_send.wait()
            if DEBUG:
                print(f"[KVCache.receive_task_info][RANK {self.rank}] Recv请求完成 - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
            return request_id


    def receive_task_info_batch(self, worker_ref_list, task_info_list): ## only support send from kvcache to worker, all tasks have the same cache worker        
        from Worker.Worker import Worker
        combined_task_info = {} ## key: infer worker
        confirmation_msg = {} ## key: request id
        for task_info in task_info_list:
            infer_worker = task_info['infer_worker']
            if infer_worker !=0 and DEBUG:
                print(f"[KVCache][RANK {self.rank}] infer worker is {infer_worker}") 
            cache_worker = task_info['cache_worker']
            req_id = task_info['request_id']
            token_num = task_info['token_num']
            item_id = task_info['id']
            cache_pages_list = task_info.get('cache_pages_list',[])
            if item_id == -1:
                continue
            # 到底是什么时候需要管理缓存？
            # 为什么只在RECV管理时会出现key error？
            if combined_task_info.get(infer_worker) == None:
                combined_task_info[infer_worker] = {}
            if task_info['task_type'] == SIGNAL_SEND:
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
            if task_info['task_type'] == SIGNAL_RECV:
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
                if task_type == SIGNAL_SEND:
                    if DEBUG:
                        print(f"[KVCache.receive_task_info_batch][RANK {self.rank}]{task_info}")
                        print(f"[KVCache.receive_task_info_batch][RANK {self.rank}] 执行Send请求 - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
                    remote_recv = rpc.rpc_async(
                        to=worker_ref_list[infer_worker].owner(), func=call_remote_method, 
                        args=(Worker.receive_kvcache_data_batch, worker_ref_list[infer_worker], task_info))
                    self.send_data_batch(task_info)
                    remote_recv.wait()
                    # print(f"[KVCache][RANK {self.rank}] 执行Send请求完成 - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
                    self.send_counter += 1

                elif task_type == SIGNAL_RECV:
                    cache_worker = task_info['cache_worker']
                    infer_worker = task_info['infer_worker']
                    worker_ref = worker_ref_list[infer_worker]
                    if DEBUG:
                        print(f"[KVCache.receive_task_info_batch][RANK {self.rank}] 执行Recv请求 - workerRank {2*infer_worker+2} -> cacheRank {2*cache_worker+3}")
                        print(f"[KVCache.receive_task_info_batch][RANK {self.rank}] 执行Recv请求 - workerRank {2*infer_worker+2} -> cacheRank {2*cache_worker+3}")
                    remote_send = rpc.rpc_async(
                        to=worker_ref.owner(), func=call_remote_method, 
                        args=(Worker.send_kvcache_data_batch,worker_ref, task_info))
                    self.receive_data_batch(task_info)
                    remote_send.wait()
                    now = datetime.now()
                    nowtime = now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"
                    # print(f"[KVCache][RANK {self.rank}] 执行Recv请求完成 - workerRank {2*infer_worker+2} -> cacheRank {2*cache_worker+3}")
                    self.recv_counter += 1

        return confirmation_msg
    
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
                infer_worker_port = 2*infer_worker + WORKER_offset
                infer_worker_addr = f"localhost:{self.master_port+infer_worker_port}"
                if task_type == SIGNAL_SEND:
                    if DEBUG:
                        print(f"[KVCache.receive_task_info_batch][RANK {self.rank}]{task_info}")
                        print(f"[KVCache {self.rank}] 执行Send请求 - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
                    combined_task_info_pb = self._task_info_json_to_pb(task_info)
                    with grpc.insecure_channel(infer_worker_addr) as channel:
                        stub = TaskInfo_pb2_grpc.InferWorkerServiceStub(channel)
                        remote_recv = stub.RecvKVCacheData.future(combined_task_info_pb)
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
        print(f"[KVCache {self.rank}]收到Coordinator请求 长度为{len(request.tasks)}")
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