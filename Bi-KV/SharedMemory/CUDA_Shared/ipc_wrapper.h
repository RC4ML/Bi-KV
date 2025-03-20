#pragma once
#include <torch/extension.h>
#include <semaphore.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#define CUDA_IPC_HANDLE_SIZE 64

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
    unsigned char cuda_handle[CUDA_IPC_HANDLE_SIZE];
    int tensor_dim;
    int64_t tensor_shape[MAX_DIM]; // 新增形状数组
};

static_assert(sizeof(cudaIpcMemHandle_t) <= CUDA_IPC_HANDLE_SIZE, 
              "CUDA IPC handle size exceeds allocation");

void producer_init(int device_id, const char* shm_name, size_t buffer_size);
void producer_send(torch::Tensor tensor);
void producer_cleanup();

void consumer_init(int device_id, const char* shm_name);
torch::Tensor consumer_receive();
void consumer_cleanup();