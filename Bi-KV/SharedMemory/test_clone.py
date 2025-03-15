import torch
import time

def test_clone_performance():
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误：需要CUDA支持的GPU设备")
        return
    
    try:
        # 计算1GB数据所需的元素数量（float32每个元素4字节）
        num_elements = (1024**3) // 4  # 1 GiB
        print(f"创建 {num_elements} 个元素的张量 (约1GiB)")

        # 在GPU上创建源张量
        src = torch.randn(num_elements, device='cuda')
        torch.cuda.synchronize()  # 确保创建完成

        # 预热GPU，避免首次运行的初始化时间影响测试
        for _ in range(3):
            _ = src.clone()
        torch.cuda.synchronize()

        # 执行多次拷贝取平均值
        num_iterations = 5
        start_time = time.time()
        for _ in range(num_iterations):
            cloned_tensor = src.clone()  # 执行拷贝
        torch.cuda.synchronize()  # 等待所有拷贝操作完成
        total_time = time.time() - start_time

        # 计算结果
        avg_time = total_time / num_iterations
        bandwidth = (1.0 / avg_time)  # 转换为GB/s

        print(f"测试次数：{num_iterations}")
        print(f"总耗时：{total_time:.4f}秒")
        print(f"平均单次拷贝时间：{avg_time:.6f}秒")
        print(f"带宽：{bandwidth:.2f} GB/s")

    except torch.cuda.OutOfMemoryError:
        print("错误：GPU内存不足，请释放显存后重试")
    except Exception as e:
        print(f"发生未知错误：{str(e)}")

if __name__ == "__main__":
    test_clone_performance()