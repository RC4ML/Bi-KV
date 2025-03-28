#include "ipc_wrapper.h"
#include <cuda_fp16.h>

// 新增：CUDA核函数实现
template <typename scalar_t>
__global__ void direct_copy_kernel(
    scalar_t* __restrict__ cache_data,
    scalar_t* __restrict__ shared_buffer,
    const int64_t* src_offsets,
    const int64_t* dest_offsets,
    const int64_t* page_sizes,
    int num_pages,
    int page_size
) {
    const int page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;

    const int elem_idx = threadIdx.x;
    const int64_t size = page_sizes[page_idx];
    if (elem_idx >= size) return;

    const int64_t src_start = src_offsets[page_idx];
    const int64_t dest_start = dest_offsets[page_idx];

    shared_buffer[dest_start + elem_idx] = cache_data[src_start + elem_idx];
}

// 新增：直接页面复制函数
extern "C"
void cuda_producer_copy_pages(
    torch::Tensor cache_data,
    torch::Tensor src_offsets,
    torch::Tensor dest_offsets,
    torch::Tensor page_sizes,
    SharedControl* producer_ctrl,
    void* producer_shared_mem,
    int page_size
) {
    TORCH_CHECK(cache_data.device().is_cuda(), "Cache data must be on CUDA");
    TORCH_CHECK(cache_data.is_contiguous(), "Cache data must be contiguous");

    //size_t current_pos = producer_ctrl->current_offset;
    const int num_pages = src_offsets.size(0);
    dim3 grid(num_pages);
    dim3 block(page_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cache_data.scalar_type(),
        "direct_copy",
        [&] {
            auto* shared_ptr = static_cast<scalar_t*>(producer_shared_mem);
            //const size_t buffer_capacity = producer_ctrl->buffer_size / sizeof(scalar_t);
            
            direct_copy_kernel<scalar_t><<<grid, block>>>(
                cache_data.data_ptr<scalar_t>(),
                shared_ptr,
                src_offsets.data_ptr<int64_t>(),
                dest_offsets.data_ptr<int64_t>(),
                page_sizes.data_ptr<int64_t>(),
                num_pages,
                page_size
            );
        }
    );

    sem_post(&producer_ctrl->sem_start);
}

