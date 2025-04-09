//ipc_wrapper.h
#pragma once
#include <torch/extension.h>
#include <semaphore.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#define CUDA_IPC_HANDLE_SIZE 64
#define MAX_DIM 5

struct SharedControl {
    sem_t sem_start;
    sem_t sem_complete;
    int data_ready;
    int error_code;
    int device_id;
    char hostname[256];
    size_t data_size;
    size_t tensor_dim;
    size_t current_offset;       // 新增：当前写入偏移量
    size_t last_valid_offset;   // 新增：最后有效数据起始位置
    size_t buffer_size;   
    unsigned char cuda_handle[CUDA_IPC_HANDLE_SIZE];
    int64_t tensor_shape[MAX_DIM]; // 新增形状数组
};

static_assert(sizeof(cudaIpcMemHandle_t) <= CUDA_IPC_HANDLE_SIZE, 
              "CUDA IPC handle size exceeds allocation");

void producer_init(int device_id, const char* shm_name);
void producer_send(torch::Tensor tensor);
//void producer_write(torch::Tensor tensor);
void producer_cleanup();
void consumer_init(int device_id, const char* shm_name, size_t buffer_size);
torch::Tensor consumer_receive();
void consumer_cleanup();
// 新增结构体定义张量维度
struct TensorDims {
    int64_t total_tokens;
    int64_t head_size;
    int64_t num_kv_heads;
    int64_t num_layers;
    int64_t kv_pair;
};

void producer_copy_pages(
    torch::Tensor cache_data,
    torch::Tensor src_offsets,
    torch::Tensor dest_offsets,
    int page_size,
    const TensorDims& dims  // 直接传递维度结构体
);

#ifdef __cplusplus
extern "C" {
#endif

void cuda_producer_copy_pages(
    torch::Tensor cache_data,
    torch::Tensor src_offsets,
    torch::Tensor dest_offsets,
    SharedControl* producer_ctrl,
    void* producer_shared_mem,
    int page_size,
    const TensorDims& dims  // 新增维度参数
);
#ifdef __cplusplus
}
#endif