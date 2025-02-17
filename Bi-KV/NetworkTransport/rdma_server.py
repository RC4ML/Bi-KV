#!/usr/bin/env python3
import rdma_transport as rdma
import time
import numpy as np
import torch
def server_main():
    # 创建 RDMAEndpoint 对象，模式设置为 "server"，监听本机所有地址，端口 7471
    ep = rdma.RDMAEndpoint("10.0.0.4", "7471", "server")
    
    buffer_size = 1024 * 1024 * 1024
    if ep.run_server() != 0:
        print("服务器启动失败")
        return
    if ep.register_memory(buffer_size) != 0:
        print("内存注册失败")
        return
    print("服务器：连接已建立")
    
    # 设定测试参数
    iterations = 10
    msg_size = 1024 * 1024 * 1024  # 每次传输 1GB 数据
    buffer = ep.get_buffer_tensor()
    # print(buffer)
    start = time.time()
    
    for i in range(iterations):
        # 投递接收请求：等待接收客户端发送的数据
        if ep.post_receive() != 0:
            print("投递接收请求失败")
            return
        if ep.poll_completion() != 0:
            print("接收轮询失败")
            return
        
        # 获取接收到的数据并验证
        # buffer = ep.get_buffer_tensor()
        # print(buffer)
        # count_ones = (buffer == 1).sum().item()
        # print(f"Number of ones: {count_ones}")
        # buffer = ep.get_buffer()
        # received_data = np.frombuffer(buffer, dtype=np.uint8, count=msg_size)
        # expected_data = np.arange(msg_size, dtype=np.uint8) % 256  # 预期的数据模式
        # if not np.array_equal(received_data, expected_data):
        #     print(f"服务器：第 {i+1} 次接收数据校验失败")
        #     return
        # else:
        #     print(f"服务器：第 {i+1} 次接收数据校验成功")

        # # 发送回复数据
        # if ep.post_send(msg_size) != 0:
        #     print("发送回复失败")
        #     return
        # if ep.poll_completion() != 0:
        #     print("发送回复轮询失败")
        #     return
    end = time.time()
    
    total_time = end - start
    print(f"服务器：共处理 {iterations} 次往返，耗时 {total_time:.6f} 秒")

if __name__ == "__main__":
    server_main()