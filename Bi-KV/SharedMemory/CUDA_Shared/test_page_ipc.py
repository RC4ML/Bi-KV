import torch
import time
import multiprocessing
import ipc_service
import os

# 模型参数和token结构定义
model_params = {
    "head_size": 128,
    "num_kv_heads": 2,
    "num_layers": 28
}

# 计算单个token的元素数量
token_elements = 1
for dim in (model_params['head_size'],
            model_params['num_kv_heads'],
            model_params['num_layers'],
            2):  # key和value
    token_elements *= dim  # 128 * 2 * 28 * 2=14336

def producer_process(device_id, shm_name, num_transfers):
    try:
        # 初始化统计
        init_start = time.perf_counter()
        ipc_service.producer_init(device_id, shm_name)
        init_end = time.perf_counter()
        print(f"[Producer] Init time: {(init_end-init_start)*1000:.2f}ms")

        # 页面参数定义
        page_size = 50               # 每页50个token
        page_elements = page_size * token_elements  # 716800 elements/page
        num_pages = 100              # 总页数
        total_elements = num_pages * page_elements

        # 构造测试数据（全1张量）
        cache_data = torch.full((total_elements,), 7, dtype=torch.float16, device='cuda')
        
        # 生成页面偏移量
        src_offsets = torch.tensor(
            [i * page_elements for i in range(num_pages)],
            dtype=torch.int64, device='cuda'
        )
        dest_offsets = torch.tensor(
            [i * page_elements for i in range(num_pages)],
            dtype=torch.int64, device='cuda'
        )

        # 预热传输（不统计）
        ipc_service.producer_copy_pages(cache_data, src_offsets, dest_offsets, page_elements)

        # 正式测试循环
        transfer_times = []
        for _ in range(num_transfers - 1):
            transfer_start = time.perf_counter()
            ipc_service.producer_copy_pages(cache_data, src_offsets, dest_offsets, page_elements)
            transfer_end = time.perf_counter()
            transfer_times.append(transfer_end - transfer_start)
        
        # 性能统计
        actual_transfers = num_transfers - 1
        total_bytes = (cache_data.numel() * cache_data.element_size() ) / (1024**2)  # MB
        total_transfer_time = sum(transfer_times)
        
        print(f"\n[Producer {device_id} Statistics]")
        print(f"  Page Size (tokens):    {page_size}")
        print(f"  Elements per Page:     {page_elements}")
        print(f"  Pages num per Transfer :{num_pages}")
        print(f"  Data per Transfer:     {total_bytes}MB")
        print(f"  Avg Transfer Time:     {sum(transfer_times)/actual_transfers*1000:.2f}ms")
        print(f"  Throughput:            {total_bytes / total_transfer_time:.2f} MB/s")

        ipc_service.producer_cleanup()
    except Exception as e:
        print(f"[Producer Error] {str(e)}")

def consumer_process(device_id, shm_name, buffer_size, num_transfers):
    try:
        # 初始化统计
        init_start = time.perf_counter()
        ipc_service.consumer_init(device_id, shm_name, buffer_size)
        init_end = time.perf_counter()
        print(f"[Consumer] Init time: {(init_end-init_start)*1000:.2f}ms")

        # 预热（丢弃首轮数据）
        _ = ipc_service.consumer_receive()

        # 预期数据参数（必须与生产者一致）
        page_size = 50
        page_elements = page_size * token_elements
        num_pages = 100
        total_elements = num_pages * page_elements
        expected = torch.full((total_elements,), 7, dtype=torch.float16, device='cuda')

        # 接收循环
        transfer_times = []
        for i in range(num_transfers - 1):
            transfer_start = time.perf_counter()
            result = ipc_service.consumer_receive()
            transfer_end = time.perf_counter()
            transfer_times.append(transfer_end - transfer_start)
            
            # 数据验证
            assert result.dim() == 1, f"Expected 1D tensor, got {result.dim()}D"
            assert result.shape[0] == total_elements, \
                f"Shape mismatch: {result.shape} vs ({total_elements},)"
                
            # 逐页验证
            for p in range(num_pages):
                page_start = p * page_elements
                page_end = page_start + page_elements
                page_data = result[page_start:page_end]
                assert torch.allclose(page_data, torch.full_like(page_data, 7)), \
                    f"Page {p} data corrupted"
            
            print(f"Transfer {i+1}/{num_transfers-1} verified")

        # 性能统计
        actual_transfers = num_transfers - 1
        total_bytes = (expected.numel() * expected.element_size()
                      * actual_transfers) / (1024*1024)  # MB
        total_transfer_time = sum(transfer_times)
        
        print(f"\n[Consumer {device_id} Statistics]")
        print(f"  Avg Receive Time:     {sum(transfer_times)/actual_transfers*1000:.2f}ms")
        print(f"  Throughput:            {total_bytes / total_transfer_time:.2f} MB/s")

        ipc_service.consumer_cleanup()
    except Exception as e:
        print(f"[Consumer Error] {str(e)}")
    finally:
        os._exit(0)

if __name__ == "__main__":
    config = {
        "device_id": 1,
        "shm_name": "/cuda_ipc_test",
        "buffer_size": 2 * 1024*1024*1024,  # 2GB缓冲区
        "num_transfers": 11          # 1次预热 + 10次实测
    }

    # 启动测试
    print(f"\n[System] Starting test with config:")
    print(f"  Device ID:     {config['device_id']}")
    print(f"  Buffer Size:   {config['buffer_size']/(1024*1024*1024):.1f}GB")
    print(f"  Total Transfers: {config['num_transfers']} (1 warmup + 100 measured)")

    system_start = time.perf_counter()

    # 启动消费者进程
    consumer = multiprocessing.Process(
        target=consumer_process,
        args=(config["device_id"], config["shm_name"], config["buffer_size"], config["num_transfers"])
    )
    consumer.start()
    time.sleep(5)  # 等待消费者初始化

    # 启动生产者进程
    producer = multiprocessing.Process(
        target=producer_process,
        args=(config["device_id"], config["shm_name"], config["num_transfers"])
    )
    producer.start()

    # 等待进程结束
    producer.join()
    consumer.join()

    total_time = time.perf_counter() - system_start
    print(f"\n[System] Total Execution Time: {total_time:.2f}s ({total_time*1000:.2f}ms)")