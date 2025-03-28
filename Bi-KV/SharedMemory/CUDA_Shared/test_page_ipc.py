import torch
import time
import multiprocessing
import ipc_service
import os

def producer_process(device_id, shm_name, num_transfers):
    try:
        init_start = time.perf_counter()
        ipc_service.producer_init(device_id, shm_name)
        init_end = time.perf_counter()
        init_time = init_end - init_start

        # 构造测试数据参数
        num_pages = 10
        page_size =5
        total_elements = num_pages * page_size
        cache_data = torch.ones(total_elements, dtype=torch.float16, device='cuda')
        print(cache_data)
        src_offsets = torch.tensor([i * page_size for i in range(num_pages)], dtype=torch.int64, device='cuda')
        dest_offsets = torch.tensor([i * page_size for i in range(num_pages)], dtype=torch.int64, device='cuda')
        page_sizes = torch.full((num_pages,), page_size, dtype=torch.int64, device='cuda')

        # 预热（不计入统计）
        ipc_service.producer_copy_pages(cache_data, src_offsets, dest_offsets, page_sizes, page_size)

        # 正式测试循环
        transfer_times = []
        for _ in range(num_transfers - 1):
            transfer_start = time.perf_counter()
            ipc_service.producer_copy_pages(cache_data, src_offsets, dest_offsets, page_sizes, page_size)
            transfer_end = time.perf_counter()
            transfer_times.append(transfer_end - transfer_start)
        
        # 统计信息
        actual_transfers = num_transfers - 1
        total_bytes = cache_data.numel() * cache_data.element_size() * actual_transfers / (1024**2)
        total_transfer_time = sum(transfer_times)
        
        print(f"\n[Producer {device_id} Statistics]")
        print(f"total_bytes:{total_bytes}")
        print(f"  Actual Transfers:  {actual_transfers}")
        print(f"  Avg Write Time:    {sum(transfer_times)/actual_transfers*1000:.2f}ms")
        print(f"  Throughput:        {total_bytes / total_transfer_time:.2f} MB/s")

        ipc_service.producer_cleanup()
    except Exception as e:
        print(f"[Producer Error] {str(e)}")

def consumer_process(device_id, shm_name, buffer_size, num_transfers):
    try:
        init_start = time.perf_counter()
        ipc_service.consumer_init(device_id, shm_name, buffer_size)
        init_end = time.perf_counter()
        init_time = init_end - init_start

        # 预热（丢弃首轮数据）
        _ = ipc_service.consumer_receive()

        # 预期数据（与生产者构造方式一致）
        num_pages = 10
        page_size =5
        total_elements = num_pages * page_size
        expected = torch.ones(total_elements, dtype=torch.float16, device='cuda')

        transfer_times = []
        for _ in range(num_transfers - 1):
            transfer_start = time.perf_counter()
            result = ipc_service.consumer_receive()
            transfer_end = time.perf_counter()
            transfer_times.append(transfer_end - transfer_start)
            print(result)
            
            # 验证数据正确性
            assert result.dim() == 1, f"Expected 1D tensor, got {result.dim()}D"
            assert result.shape[0] == total_elements, f"Expected shape ({total_elements},), got {result.shape}"
            assert torch.allclose(result, expected), "Data mismatch"
            print(f"Received tensor with shape {result.shape} verified.")

        # 统计信息
        actual_transfers = num_transfers - 1
        total_bytes = expected.numel() * expected.element_size() * actual_transfers / (1024**2)
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
        "buffer_size": 1024 * 1024 * 1024,  # 1GB缓冲区
        "num_transfers": 11                  # 1次预热 + 10次实测
    } 

    system_start = time.perf_counter()

    consumer = multiprocessing.Process(
        target=consumer_process,
        args=(config["device_id"], config["shm_name"], config["buffer_size"], config["num_transfers"])
    )
    producer = multiprocessing.Process(
        target=producer_process,
        args=(config["device_id"], config["shm_name"], config["num_transfers"])
    )
    
    consumer.start()
    time.sleep(5)  # 等待消费者初始化
    producer.start()

    producer.join()
    consumer.join()

    total_time = time.perf_counter() - system_start
    print(f"\n[System] Total Execution Time: {total_time * 1000:.2f}ms")