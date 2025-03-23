import torch
import time
import multiprocessing
import ipc_service
import os

def producer_process(device_id, shm_name ,num_transfers):
    try:
        # 初始化计时
        init_start = time.perf_counter()
        ipc_service.producer_init(device_id, shm_name)
        init_end = time.perf_counter()
        init_time = init_end - init_start

        # 首轮预热（不计入统计）
        src = torch.ones([3530, 128, 2, 28, 2], device='cuda', dtype=torch.float16)
        ipc_service.producer_send(src)  # 第一轮仅预热
        
        # 正式测试循环
        transfer_times = []
        for _ in range(num_transfers - 1):  # 减少一次循环
            transfer_start = time.perf_counter()
            ipc_service.producer_send(src)
            transfer_end = time.perf_counter()
            transfer_times.append(transfer_end - transfer_start)
            
        # 统计计算（使用实际测试次数）
        actual_transfers = num_transfers - 1
        total_bytes = src.numel() * src.element_size() * actual_transfers / (1024**2)
        total_transfer_time = sum(transfer_times)
        
        print(f"\n[Producer {device_id} Statistics]")
        print(f"  Actual Transfers:  {actual_transfers}")
        print(f"  Avg Write Time:    {sum(transfer_times)/actual_transfers*1000:.2f}ms")
        print(f"  Throughput:        {total_bytes / total_transfer_time:.2f} MB/s")

        ipc_service.producer_cleanup()
    except Exception as e:
        print(f"[Producer Error] {str(e)}")

def consumer_process(device_id, shm_name, buffer_size,num_transfers):
    try:
        # 初始化计时
        init_start = time.perf_counter()
        ipc_service.consumer_init(device_id, shm_name,buffer_size)
        init_end = time.perf_counter()
        init_time = init_end - init_start

        # 首轮预热（不计入统计）
        _ = ipc_service.consumer_receive()  # 丢弃第一轮数据
        
        # 正式测试循环
        transfer_times = []
        for _ in range(num_transfers - 1):  # 减少一次循环
            transfer_start = time.perf_counter()
            result = ipc_service.consumer_receive()
            print(result.shape)
            transfer_end = time.perf_counter()
            transfer_times.append(transfer_end - transfer_start)

        # 统计计算（使用实际测试次数）
        actual_transfers = num_transfers - 1
        total_bytes = result.numel() * result.element_size() * actual_transfers / (1024**2)
        total_transfer_time = sum(transfer_times)
        
        print(f"\n[Consumer {device_id} Statistics]")
        print(f"  Actual Transfers:  {actual_transfers}")
        print(f"  Avg Read Time:     {sum(transfer_times)/actual_transfers*1000:.2f}ms")
        print(f"  Throughput:        {total_bytes / total_transfer_time:.2f} MB/s")

        ipc_service.consumer_cleanup()
    except Exception as e:
        print(f"[Consumer Error] {str(e)}")
    finally:
        os._exit(0)

if __name__ == "__main__":
    config = {
        "device_id": 0,
        "shm_name": "/cuda_ipc_test",
        "buffer_size": 1024 * 1024 * 1024,  # 1GB
        "num_transfers": 11                # 包含1次预热+10次实测
    } 

    # 系统级计时（排除初始化时间）
    system_start = time.perf_counter()

    consumer = multiprocessing.Process(
        target=consumer_process,
        args=(config["device_id"], config["shm_name"], config["buffer_size"],config["num_transfers"])
    )
    producer = multiprocessing.Process(
        target=producer_process,
        args=(config["device_id"], config["shm_name"], 
            config["num_transfers"])
    )
    
    consumer.start()
    time.sleep(5)  # 缩短等待时间
    producer.start()

    producer.join()
    consumer.join()

    # 系统级统计（仅计算有效传输时间）
    total_time = time.perf_counter() - system_start
    print(f"\n[System] Total Execution Time: {total_time * 1000:.2f}ms")