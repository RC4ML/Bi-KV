import torch
import time
import multiprocessing
import ipc_service
import os

def producer_process(device_id, shm_name, buffer_size, num_transfers, data_size):
    try:
        # 初始化计时
        init_start = time.perf_counter()
        ipc_service.producer_init(device_id, shm_name, buffer_size)
        init_end = time.perf_counter()
        init_time = init_end - init_start

        # 数据传输计时
        transfer_times = []
        for _ in range(num_transfers):
            # src = torch.ones((data_size,), 
            #                 dtype=torch.float16,
            #                 device=f'cuda:{device_id}')
            src = torch.ones([3530, 128, 2, 28, 2], device='cuda', dtype=torch.float16)
            # 仅测量发送操作时间
            transfer_start = time.perf_counter()
            ipc_service.producer_send(src)
            transfer_end = time.perf_counter()
            transfer_times.append(transfer_end - transfer_start)
        total_bytes = src.numel() * src.element_size()/ (1024**2)  # 正确计算总字节
        # 计算统计指标
        total_transfer_time = sum(transfer_times)
        total_time = init_time + total_transfer_time
        
        print(f"\n[Producer {device_id} Statistics]")
        print(f"  Init Time:         {init_time * 1000:.2f}ms")
        print(f"  Avg Write Time:    {sum(transfer_times)/len(transfer_times)*1000:.2f}ms")
        print(f"  Total Write Time:  {total_transfer_time * 1000:.2f}ms")
        print(f"  Throughput (excl init): {total_bytes / total_transfer_time:.2f} MB/s")
        print(f"  Throughput (incl init): {total_bytes / total_time:.2f} MB/s")

        ipc_service.producer_cleanup()
    except Exception as e:
        print(f"[Producer Error] {str(e)}")

def consumer_process(device_id, shm_name, num_transfers, expected_size):
    try:
        # 初始化计时
        init_start = time.perf_counter()
        ipc_service.consumer_init(device_id, shm_name)
        init_end = time.perf_counter()
        init_time = init_end - init_start

        # 数据传输计时
        transfer_times = []
        for _ in range(num_transfers):
            # 仅测量接收操作时间
            transfer_start = time.perf_counter()
            result = ipc_service.consumer_receive()
            transfer_end = time.perf_counter()
            transfer_times.append(transfer_end - transfer_start)
            
            # # 数据验证
            # assert result.shape[0] == expected_size, "Tensor size mismatch"
            # assert torch.allclose(result, torch.ones_like(result), atol=1e-3), "Data validation failed"

        # 计算统计指标
        total_transfer_time = sum(transfer_times)
        total_time = init_time + total_transfer_time
        total_bytes =num_transfers*result.numel() * result.element_size()/ (1024**2)  # 正确计算总字节
        print(f"result.size{result.size()}")
        #total_data = (expected_size * 2 * num_transfers) / (1024**2)  # MB
        print(f"\n[Consumer {device_id} Statistics]")
        print(f"[Consumer]total:{total_bytes}MB \n")
        print(f" [Consumer] Init Time:         {init_time * 1000:.2f}ms")
        print(f" [Consumer] Avg Read Time:     {sum(transfer_times)/len(transfer_times)*1000:.2f}ms")
        print(f" [Consumer] Total Read Time:   {total_transfer_time * 1000:.2f}ms")
        print(f" [Consumer] Throughput (excl init): {total_bytes / total_transfer_time:.2f} MB/s")
        print(f" [Consumer] Throughput (incl init): {total_bytes / total_time:.2f} MB/s")

        ipc_service.consumer_cleanup()
    except Exception as e:
        print(f"[Consumer Error] {str(e)}")
    finally:
        os._exit(0)

if __name__ == "__main__":
    config = {
        "device_id": 0,
        "shm_name": "/cuda_ipc_test",
        "buffer_size": 1024*1024 * 1024,  # 1GB
        "data_size": 100 * 1024 * 1024,       # 10MB as half floats (5M elements)
        "num_transfers": 10             # total 100MB
    } 

    # 系统级计时
    system_start = time.perf_counter()

    producer = multiprocessing.Process(
        target=producer_process,
        args=(config["device_id"], config["shm_name"], 
              config["buffer_size"], config["num_transfers"],
              config["data_size"])
    )
    consumer = multiprocessing.Process(
        target=consumer_process,
        args=(config["device_id"], config["shm_name"],
              config["num_transfers"], config["data_size"])
    )

    producer.start()
    time.sleep(5)  # 延长等待时间确保初始化完成
    consumer.start()

    producer.join()
    consumer.join()

    # 系统级统计
    total_time = time.perf_counter() - system_start
    total_data = (config["data_size"] * 4 * config["num_transfers"]) / (1024**2)
    
    print(f"\n[System Statistics]")
    print(f"  Total Execution Time: {total_time * 1000:.2f}ms")
    print(f"  System Throughput:    {total_data / total_time:.2f} MB/s")