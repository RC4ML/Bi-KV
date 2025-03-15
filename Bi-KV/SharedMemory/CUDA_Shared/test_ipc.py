import torch
import time
import multiprocessing
import ipc_service
import os

def producer_process(device_id, shm_name, data_size):
    try:
        # 初始化数据张量
        src = torch.ones((data_size,), 
                        dtype=torch.float32,
                        device=f'cuda:{device_id}')
        
        # 启动生产者
        ipc_service.producer(device_id, shm_name, src)
        # print(f"[Producer {device_id}] Sent {data_size} elements")
    except Exception as e:
        print(f"[Producer Error] {str(e)}")

def consumer_process(device_id, shm_name, expected_size):
    try:
        start_time = time.time()
        
        # 接收数据
        result = ipc_service.consumer(device_id, shm_name)
        # print(result)
        latency = time.time() - start_time
        
        # 数据验证
        assert result.shape[0] == expected_size, "Tensor size mismatch"
        assert torch.allclose(result, torch.ones_like(result)), "Data validation failed"
        
        # 性能统计
        data_bytes = result.element_size() * result.numel()
        throughput = (data_bytes / 1024**2) / latency  # GB/s
        # print(f"[Consumer {device_id}] Validation success | "
              f"Latency: {latency*1000:.2f}ms | "
              f"Throughput: {throughput:.2f} MB/s")
    except Exception as e:
        print(f"[Consumer Error] {str(e)}")
    finally:
        ipc_service.cleanup(shm_name)
        os._exit(0)  # 确保进程退出  # 新增代码

if __name__ == "__main__":
    config = {
        "device_id": 0,
        "shm_name": "/cuda_ipc_test",
        "data_size": 100 * 1024 * 1024  # 10M elements
    }

    # 创建进程
    producer = multiprocessing.Process(
        target=producer_process,
        args=(config["device_id"], config["shm_name"], config["data_size"])
    )
    consumer = multiprocessing.Process(
        target=consumer_process,
        args=(config["device_id"], config["shm_name"], config["data_size"])
    )

    # 执行测试
    system_start = time.time()
    producer.start()
    time.sleep(0.5)  # 确保生产者先初始化共享内存
    consumer.start()

    producer.join()
    consumer.join()

    # 系统级统计
    total_time = time.time() - system_start
    print(f"\n[System] Total execution time: {total_time*1000:.2f}ms")