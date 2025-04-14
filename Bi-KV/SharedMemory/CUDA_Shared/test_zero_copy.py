import torch
import time
import multiprocessing
import ipc_service
import os

# 模型参数配置
model_params = {
    "head_size": 128,
    "num_kv_heads": 2,
    "num_layers": 28,
    "kv_pair": 2
}

def create_tensor_dims(page_size, num_pages):
    dims = ipc_service.TensorDims()
    dims.total_tokens = num_pages * page_size
    dims.head_size = model_params["head_size"]
    dims.num_kv_heads = model_params["num_kv_heads"]
    dims.num_layers = model_params["num_layers"]
    dims.kv_pair = model_params["kv_pair"]
    return dims

def calculate_token_elements():
    return model_params['head_size'] * model_params['num_kv_heads'] * model_params['num_layers'] * model_params['kv_pair']

def producer_process(device_id, shm_name, num_transfers):
    try:
        # 初始化
        ipc_service.producer_init(device_id, shm_name)
        
        # 页面参数
        page_size = 50
        num_pages = 200
        token_elements = calculate_token_elements()
        page_elements = page_size * token_elements
        dims = create_tensor_dims(page_size, num_pages)

        # 准备固定内存的CPU数据
        cpu_data = torch.full(
            (num_pages * page_size * token_elements,),
            7, dtype=torch.float16, device='cpu'
        ).pin_memory()

        # 准备页面索引和偏移量
        page_indices = torch.arange(num_pages, dtype=torch.int64, device='cuda')
        dest_offsets = torch.arange(0, num_pages * page_elements, page_elements, 
                                  dtype=torch.int64, device='cuda')

        # 预热
        ipc_service.producer_zero_copy_pages(
            cpu_data, page_indices, dest_offsets, page_size, dims)

        # 性能测试
        transfer_times = []
        for _ in range(num_transfers - 1):
            start = time.perf_counter()
            ipc_service.producer_zero_copy_pages(
                cpu_data, page_indices, dest_offsets, page_size, dims)
            transfer_times.append(time.perf_counter() - start)

        # 打印统计信息
        data_size = cpu_data.numel() * cpu_data.element_size() / (1024**2) # MB
        avg_time = sum(transfer_times) / len(transfer_times)
        
        print(f"\n[Zero Copy Producer Stats]")
        print(f"  Page Size: {page_size} tokens")
        print(f"  Pages per Transfer: {num_pages}")
        print(f"  Data per Transfer: {data_size:.2f}MB")
        print(f"  Avg Transfer Time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {data_size / avg_time:.2f} MB/s")

        ipc_service.producer_cleanup()
    except Exception as e:
        print(f"Producer error: {str(e)}")

def consumer_process(device_id, shm_name, buffer_size, num_transfers):
    try:
        # 初始化
        ipc_service.consumer_init(device_id, shm_name, buffer_size)
        
        # 页面参数（必须与生产者匹配）
        page_size = 50
        num_pages = 200
        dims = create_tensor_dims(page_size, num_pages)
        expected_shape = (
            num_pages * page_size,
            dims.head_size,
            dims.num_kv_heads,
            dims.num_layers,
            dims.kv_pair
        )

        # 接收数据
        transfer_times = []
        for _ in range(num_transfers - 1):
            start = time.perf_counter()
            tensor = ipc_service.consumer_receive()
            transfer_times.append(time.perf_counter() - start)
            
            # 验证数据
            assert tensor.shape == expected_shape
            assert torch.all(tensor == 7)
            print(tensor.device)

        # 打印统计信息
        data_size = tensor.numel() * tensor.element_size() / (1024**2) # MB
        avg_time = sum(transfer_times) / len(transfer_times)
        
        print(f"\n[Consumer Stats]")
        print(f"  Avg Receive Time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {data_size / avg_time:.2f} MB/s")

        ipc_service.consumer_cleanup()
    except Exception as e:
        print(f"Consumer error: {str(e)}")
    finally:
        os._exit(0)

if __name__ == "__main__":
    config = {
        "device_id": 0,
        "shm_name": "/cuda_zero_copy_test",
        "buffer_size": 2 * 1024**3,  # 2GB
        "num_transfers": 11           # 1 warmup + 10 measured
    }

    print("Starting Zero Copy IPC test...")
    
    # 启动消费者
    consumer = multiprocessing.Process(
        target=consumer_process,
        args=(config["device_id"], config["shm_name"], 
              config["buffer_size"], config["num_transfers"])
    )
    consumer.start()
    time.sleep(2)  # 等待消费者初始化

    # 启动生产者
    producer = multiprocessing.Process(
        target=producer_process,
        args=(config["device_id"], config["shm_name"], config["num_transfers"])
    )
    producer.start()

    # 等待完成
    producer.join()
    consumer.join()
    print("Test completed")