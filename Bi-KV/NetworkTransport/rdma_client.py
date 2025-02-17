#!/usr/bin/env python3
import rdma_transport as rdma
import time
import numpy as np

def client_main():
    # 修改下面 server_ip 为服务器的实际 IP 地址
    server_ip = "10.0.0.4"
    port = "7471"
    ep = rdma.RDMAEndpoint(server_ip, port, "client")
    
    buffer_size = 1024 * 1024 * 1024
    if ep.connect_client() != 0:
        print("客户端连接失败")
        return
    if ep.register_memory(buffer_size) != 0:
        print("内存注册失败")
        return
    print("客户端：连接已建立")
    
    iterations = 10
    msg_size = 1024 * 1024 * 1024  # 每次发送 1GB 数据
    start = time.time()
    for i in range(iterations):
        # 填充发送数据
        # buffer = ep.get_buffer_tensor()
        # print(buffer)
        
        # # 获取 buffer 的底层内存地址
        # buffer_address = buffer.__array_interface__['data'][0]
        # print(f"客户端：第 {i+1} 次迭代，buffer 地址: {hex(buffer_address)}, 实际大小: {msg_size} 字节")
        # # 确保 msg_size 不超过 buffer 的实际大小
        # actual_size = min(msg_size, buffer_size)
        # send_data = np.arange(actual_size, dtype=np.uint8) % 256  # 填充递增的字节序列
        # np.copyto(buffer[:actual_size], send_data)
        
        # # 投递接收请求，用于接收服务器回复
        # if ep.post_receive() != 0:
        #     print("投递接收请求失败")
        #     return
        
        # 发送数据
        if ep.post_send(msg_size) != 0:
            print("投递发送请求失败")
            return
        if ep.poll_completion() != 0:
            print("发送轮询失败")
            return
        
        # # 接收服务器回复并验证
        # if ep.poll_completion() != 0:
        #     print("接收轮询失败")
        #     return
        # received_data = np.frombuffer(buffer, dtype=np.uint8, count=msg_size)
        # if not np.array_equal(received_data, send_data):
        #     print(f"客户端：第 {i+1} 次接收数据校验失败")
        #     return
        # else:
        #     print(f"客户端：第 {i+1} 次接收数据校验成功")
    end = time.time()
    
    total_time = end - start
    # 每次往返传输 2*msg_size 字节数据
    total_bytes = iterations * msg_size
    throughput = total_bytes * 8 / total_time / (1024 * 1024 * 1024)  # Gbps
    avg_latency_ms = (total_time / iterations) * 1000  # 毫秒/往返
    
    print(f"客户端：共 {iterations} 次发送，耗时 {total_time:.6f} 秒")
    # print(f"平均延时: {avg_latency_ms:.3f} ms")
    print(f"吞吐量: {throughput:.3f} Gbps")
    
if __name__ == "__main__":
    client_main()