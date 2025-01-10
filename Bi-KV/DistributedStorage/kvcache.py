from ast import Dict
from curses.ascii import SI
import random

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from DistributedStorage.Signals import SIGNAL_SEND, SIGNAL_RECV, SIGNAL_ACK, SIGNAL_TERMINATE
from rpc_def import KVCACHE_offset,WORKER_offset
from Remote.remote_call import call_remote_method
from config import *
import time
# TODO 提取model_params作为公共参数
model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

token_shape = (model_params['head_size'],
               model_params['num_kv_heads'],
               model_params['num_layers'],
               2)

class KVCache:
    def __init__(self, rank):
        self.rank = rank + KVCACHE_offset
        self.cpu_index = rank
        self.cache_size = 100000
        self.cache_control_dict = {}
        self.cache_data = torch.full(
            (self.cache_size,) + token_shape, 
            self.rank,
            device='cpu',
            dtype=torch.float16
        )
        self.start_pos = 0
        print(f"[KVCache][CPU index:{rank} rank: {self.rank}] 初始化：Tensor大小={self.cache_data.size()}，值={self.rank}")

    def send_data(self,task_info:Dict):
        dst_rank = task_info['infer_worker'] + WORKER_offset
        request_id = task_info['request_id']
        token_num = task_info['token_num']
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={token_num}")
        # TODO 实际的读写逻辑大概不是这样
        start_pos = random.randint(0,self.cache_size/2)
        send_tensor = self.cache_data[start_pos:start_pos+token_num]
        dist.send(tensor=send_tensor, dst=dst_rank)
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 请求ID={request_id}, 长度={token_num}")

    def send_data_batch(self,combined_task_info:Dict):
        dst_rank = combined_task_info['infer_worker'] + WORKER_offset
        token_num = combined_task_info['token_num']
        id_token_pair_list = combined_task_info['id_token_pair']
        send_tensor_list = []
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 长度={token_num}")
        for id_token_pair in id_token_pair_list:
            item_id = id_token_pair[0]
            if item_id not in self.cache_control_dict:
                pass
            start_pos,next_pos = self.cache_control_dict[item_id]
            if next_pos-start_pos != id_token_pair[1]:
                print(f"[KVCache][Rank {self.rank}] Fatal Error: offest({next_pos-start_pos}) != token num({id_token_pair[1]}) id index{id_token_pair_list.index(id_token_pair)}")
                print(f"[KVCache][Rank {self.rank}] item_id={item_id}, start_pos={start_pos}, next_pos={next_pos} id_token_pair={id_token_pair}")  
                print(f"[KVCache][Rank {self.rank}] To continue the process, we have to ignore this error")
                next_pos = start_pos + id_token_pair[1]
            send_tensor_list.append(self.cache_data[start_pos:next_pos])
        send_tensor = torch.cat(send_tensor_list, dim=0)
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
        src_rank = infer_worker + WORKER_offset
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 开始接收数据从 Rank {src_rank}")
        recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
        )
        dist.recv(tensor=recv_tensor, src=src_rank)
        if DEBUG:
            print(f"[KVCache][CPU {self.cpu_index}] [rank{self.rank}] 完成接收数据从 Rank {infer_worker} [rank{src_rank}]")
        next_pos = self.start_pos + token_num
        if next_pos > self.cache_size:
            self.start_pos = 0
            next_pos = self.start_pos + token_num
        # self.cache_data[self.start_pos:next_pos] = recv_tensor
        self.cache_control_dict[item_id] = (self.start_pos,next_pos)
        self.start_pos = next_pos

    def receive_data_batch(self, combined_task_info:Dict):
        # request_id = task_info['request_id']
        infer_worker = combined_task_info['infer_worker']
        token_num = combined_task_info['token_num']
        id_token_pair_list = combined_task_info['id_token_pair']
        src_rank = infer_worker + WORKER_offset
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 开始接收数据从 Rank {src_rank} 长度为{token_num}")
        recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
        )
        dist.recv(tensor=recv_tensor, src=src_rank)
        if DEBUG:
            print(f"[KVCache][CPU {self.cpu_index}] [rank{self.rank}] 完成接收数据从 Rank {infer_worker} [rank{src_rank}]")
        for id_token_pair in id_token_pair_list:
            self._manage_cache(id_token_pair[0], id_token_pair[1])

    def send_confirmation(self, confirmation_msg):
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 发送确认消息到调度器: 请求ID={confirmation_msg}")
        return confirmation_msg

    def terminate(self):
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 收到终止信号，退出运行")
        return "Terminated"
    
    def receive_task_info(self, task_info:Dict, worker_ref):
        from Worker.Worker import Worker
        if DEBUG:
            print(f"[KVCache][RANK {self.rank}] taskinfo is {task_info}")
        # task_type, request_id, cache_worker, infer_worker = task_info
        task_type = task_info['task_type']
        request_id = task_info['request_id']
        # return request_id
        if task_type == SIGNAL_SEND:
            remote_recv = rpc.rpc_async(
                to=worker_ref.owner(), func=call_remote_method, 
                args=(Worker.receive_kvcache_data_batch,worker_ref, task_info))
            self.send_data(task_info)
            remote_recv.wait()
            return request_id
        elif task_type == SIGNAL_RECV:
            cache_worker = task_info['cache_worker']
            infer_worker = task_info['infer_worker']
            print(f"[KVCache] 执行请求 {request_id} - Rank {infer_worker+WORKER_offset} -> Rank {cache_worker+KVCACHE_offset}")
            remote_send = rpc.rpc_async(
                to=worker_ref.owner(), func=call_remote_method, 
                args=(Worker.send_kvcache_data_batch,worker_ref, task_info))
            self.receive_data(task_info)
            remote_send.wait()
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
            if item_id == 4362 or item_id == 1884:
                print(f"[KVCache][RANK {self.rank}] {item_id} token_num = {token_num} in req {req_id}")
            if item_id == -1:
                # 似乎confirmation_msg需要考虑到-1重计算任务
                if confirmation_msg.get(req_id) == None:
                    confirmation_msg[req_id] = 1
                else:
                    confirmation_msg[req_id] += 1
                continue
            # 到底是什么时候需要管理缓存？
            # 为什么只在RECV管理时会出现key error？
            self._manage_cache(item_id, token_num)
            if combined_task_info.get(infer_worker) == None:
                combined_task_info[infer_worker] = {}
            if task_info['task_type'] == SIGNAL_SEND:
                if combined_task_info[infer_worker].get(SIGNAL_SEND) == None:
                    combined_task_info[infer_worker][SIGNAL_SEND] = {"infer_worker":infer_worker, 
                                                                "cache_worker":cache_worker,
                                                                "token_num":token_num,
                                                                'task_type': SIGNAL_SEND,
                                                                'id_token_pair':[(item_id,token_num)],
                                                                } 
                else:
                    combined_task_info[infer_worker][SIGNAL_SEND]['token_num'] += token_num
                    combined_task_info[infer_worker][SIGNAL_SEND]['id_token_pair'].append((item_id,token_num))
            if task_info['task_type'] == SIGNAL_RECV:
                # 按理说应该是只有RECV才要管理？
                # self._manage_cache(item_id, token_num)
                if combined_task_info[infer_worker].get(SIGNAL_RECV) == None:
                    combined_task_info[infer_worker][SIGNAL_RECV] = {"infer_worker":infer_worker, 
                                                                "cache_worker":cache_worker,
                                                                "token_num":token_num,
                                                                'task_type': SIGNAL_RECV,
                                                                'id_token_pair':[(item_id,token_num)],
                                                                }    
                else:
                    combined_task_info[infer_worker][SIGNAL_RECV]['token_num'] += token_num
                    combined_task_info[infer_worker][SIGNAL_RECV]['id_token_pair'].append((item_id,token_num))
            if confirmation_msg.get(req_id) == None:
                confirmation_msg[req_id] = 1
            else:
                confirmation_msg[req_id] += 1
        # 初始状态下，第一轮是全空的任务
        for task_infer_worker in combined_task_info:
            # task_info = combined_task_info[task_infer_worker]
            combined_task_list = combined_task_info[task_infer_worker]
            for task_info in combined_task_list.values():
                task_type = task_info['task_type']
                infer_worker = task_info['infer_worker']
                # print(f"[KVCache][RANK {self.rank}] task type is {task_type}")
                if task_type == SIGNAL_SEND:
                    # print(f"[KVCache][RANK {self.rank}] 执行Send请求 - Rank {cache_worker+KVCACHE_offset} -> Rank {infer_worker+WORKER_offset}")
                    remote_recv = rpc.rpc_async(
                        to=worker_ref_list[infer_worker].owner(), func=call_remote_method, 
                        args=(Worker.receive_kvcache_data_batch, worker_ref_list[infer_worker], task_info))
                    self.send_data_batch(task_info)
                    remote_recv.wait()
            
                elif task_type == SIGNAL_RECV:
                    cache_worker = task_info['cache_worker']
                    infer_worker = task_info['infer_worker']
                    worker_ref = worker_ref_list[infer_worker]
                    # print(f"[KVCache][RANK {self.rank}] 执行Recv请求 - Rank {infer_worker+WORKER_offset} -> Rank {cache_worker+KVCACHE_offset}")
                    remote_send = rpc.rpc_async(
                        to=worker_ref.owner(), func=call_remote_method, 
                        args=(Worker.send_kvcache_data_batch,worker_ref, task_info))
                    self.receive_data_batch(task_info)
                    remote_send.wait()
        return confirmation_msg
    
    def _manage_cache(self, item_id, token_num):
        if item_id in self.cache_control_dict:
            return self.cache_control_dict[item_id]
        next_pos = self.start_pos + token_num
        if next_pos > self.cache_size:
            self.start_pos = 0
            next_pos = token_num
        self.cache_control_dict[item_id] = (self.start_pos,next_pos)
        self.start_pos = next_pos
        if item_id == 4362 or item_id == 1884:
            print(f"[KVCache][Rank {self.rank}] item_id={item_id},token_num = {token_num}, start_pos={self.cache_control_dict[item_id][0]}, next_pos={self.cache_control_dict[item_id][1]}")
        return self.cache_control_dict[item_id]