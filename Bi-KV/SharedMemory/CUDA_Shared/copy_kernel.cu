//copy_kernel.cu
#include "ipc_wrapper.h"
#include <cuda_fp16.h>
// src_offsets = torch.tensor(  已经包含了pagesize偏移，不要再乘page_size了
//     [p * self.page_size for p in page_indices],
//     dtype=torch.int64,
//     device=self.device
// )
template <typename scalar_t>
__global__ void block_per_page_copy_kernel(
    scalar_t* __restrict__ cache_data,
    scalar_t* __restrict__ shared_buffer,
    const int64_t* src_offsets,
    const int64_t* dest_offsets,
    int num_pages,
    int page_size  // 新增固定页面大小参数
) {
    // 二维Grid到线性索引转换
    const int page_idx = blockIdx.x * gridDim.y + blockIdx.y;
    if (page_idx >= num_pages) return;

    const int64_t src_start = src_offsets[page_idx];
    const int64_t dest_start = dest_offsets[page_idx];
    
    // 计算每个线程需要处理的元素数
    const int elements_per_thread = (page_size + blockDim.x - 1) / blockDim.x;
    
    // 并行拷贝整页数据
    for(int i = 0; i < elements_per_thread; ++i) {
        const int elem_idx = threadIdx.x + i * blockDim.x;
        if(elem_idx < page_size) {
            shared_buffer[dest_start + elem_idx] = 
                cache_data[src_start + elem_idx];
        }
    }
}

// 基于页处理的 Zero Copy 内核
template <typename scalar_t>
__global__ void zero_copy_page_kernel(
    const scalar_t* __restrict__ cpu_data,
    scalar_t* __restrict__ shared_buffer,
    const int64_t* page_indices,
    const int64_t* dest_offsets,
    int num_pages,
    int page_elements//就是pagesize
) {
    // 二维Grid到线性索引转换
    const int page_idx = blockIdx.x * gridDim.y + blockIdx.y;
    if (page_idx >= num_pages) return;

    const int64_t src_offset = page_indices[page_idx] ;
    const int64_t dest_offset = dest_offsets[page_idx];
    
    // 每个线程处理多个元素
    const int elements_per_thread = (page_elements + blockDim.x - 1) / blockDim.x;
    
    // 使用展开循环优化
    // #pragma unroll
    for(int i = 0; i < elements_per_thread; ++i) {
        const int elem_idx = threadIdx.x + i * blockDim.x;
        if(elem_idx < page_elements) {
            shared_buffer[dest_offset + elem_idx] = cpu_data[src_offset + elem_idx];
        }
    }
}

extern "C"
void cuda_producer_copy_pages(
    torch::Tensor cache_data,
    torch::Tensor src_offsets,
    torch::Tensor dest_offsets,
    SharedControl* producer_ctrl,
    void* producer_shared_mem,
    int page_size,
    const TensorDims& dims  // 新增维度参数
) {
    TORCH_CHECK(cache_data.device().is_cuda(), "Cache data must be on CUDA");
    TORCH_CHECK(cache_data.is_contiguous(), "Cache data must be contiguous");

    const int num_pages = src_offsets.size(0);
    const int elements_per_token = dims.head_size * dims.num_kv_heads * dims.num_layers * dims.kv_pair;
    const int page_elements = page_size * elements_per_token;
    const int data_size = num_pages * page_elements * sizeof(half);
    
    // 检查缓冲区溢出
    if (producer_ctrl->current_offset + data_size > producer_ctrl->buffer_size) {
        producer_ctrl->current_offset = 0;
    }

    // 二维Grid配置
    dim3 grid((num_pages + 31) / 32, 32);
    dim3 block(1024);

    void* write_ptr = static_cast<void*>(producer_shared_mem) + producer_ctrl->current_offset;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cache_data.scalar_type(),
        "block_per_page_copy",
        [&] {
            auto* shared_ptr = static_cast<scalar_t*>(write_ptr);
            
            block_per_page_copy_kernel<scalar_t><<<grid, block>>>(
                cache_data.data_ptr<scalar_t>(),
                shared_ptr,
                src_offsets.data_ptr<int64_t>(),
                dest_offsets.data_ptr<int64_t>(),
                num_pages,
                page_elements
            );
        }
    );

    // 更新控制信息
    cudaDeviceSynchronize();
    producer_ctrl->last_valid_offset = producer_ctrl->current_offset;
    producer_ctrl->current_offset += data_size;
    
    // 设置五维张量形状
    producer_ctrl->tensor_dim = 5;
    producer_ctrl->tensor_shape[0] = num_pages * page_size;  // 总token数
    producer_ctrl->tensor_shape[1] = dims.head_size;        // 128
    producer_ctrl->tensor_shape[2] = dims.num_kv_heads;     // 2
    producer_ctrl->tensor_shape[3] = dims.num_layers;       // 28
    producer_ctrl->tensor_shape[4] = dims.kv_pair;          // 2
    
    producer_ctrl->data_size = data_size;
    sem_post(&producer_ctrl->sem_start);
}


// 基于页处理的 Zero Copy 包装函数
void cuda_producer_zero_copy_pages(
    torch::Tensor cpu_data,
    torch::Tensor page_indices,
    torch::Tensor dest_offsets,
    SharedControl* producer_ctrl,
    void* producer_shared_mem,
    int page_size,
    const TensorDims& dims
) {
    TORCH_CHECK(cpu_data.is_pinned(), "CPU data must be pinned memory");
    TORCH_CHECK(page_indices.device().is_cuda(), "Page indices must be on CUDA");
    TORCH_CHECK(dest_offsets.device().is_cuda(), "Dest offsets must be on CUDA");
    
    const int num_pages = page_indices.size(0);
    const int elements_per_token = dims.head_size * dims.num_kv_heads * dims.num_layers * dims.kv_pair;
    const int page_elements = page_size * elements_per_token;
    const int data_size = num_pages * page_elements * sizeof(half);
    
    // 检查缓冲区溢出
    if (producer_ctrl->current_offset + data_size > producer_ctrl->buffer_size) {
        producer_ctrl->current_offset = 0;
    }

    // 二维Grid配置 (优化网格布局)
    dim3 grid((num_pages + 31) / 32, 32);
    dim3 block(1024);

    void* write_ptr = static_cast<void*>(producer_shared_mem) + producer_ctrl->current_offset;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cpu_data.scalar_type(),
        "zero_copy_page_kernel",
        [&] {
            auto* shared_ptr = static_cast<scalar_t*>(write_ptr);
            
            zero_copy_page_kernel<scalar_t><<<grid, block>>>(
                cpu_data.data_ptr<scalar_t>(),
                shared_ptr,
                page_indices.data_ptr<int64_t>(),
                dest_offsets.data_ptr<int64_t>(),
                num_pages,
                page_elements
            );
            
            // C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    // 更新控制信息
    cudaDeviceSynchronize();
    producer_ctrl->last_valid_offset = producer_ctrl->current_offset;
    producer_ctrl->current_offset += data_size;
    
    // 设置五维张量形状
    producer_ctrl->tensor_dim = 5;
    producer_ctrl->tensor_shape[0] = num_pages * page_size;  // 总token数
    producer_ctrl->tensor_shape[1] = dims.head_size;       // 128
    producer_ctrl->tensor_shape[2] = dims.num_kv_heads;    // 2
    producer_ctrl->tensor_shape[3] = dims.num_layers;      // 28
    producer_ctrl->tensor_shape[4] = dims.kv_pair;         // 2
    
    producer_ctrl->data_size = data_size;
    sem_post(&producer_ctrl->sem_start);
}