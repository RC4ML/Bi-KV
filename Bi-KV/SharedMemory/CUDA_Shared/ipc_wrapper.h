#pragma once
#include <torch/extension.h>
#include <semaphore.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>

#define CUDA_IPC_HANDLE_SIZE 64  // CUDA规范定义

struct SharedControl {
    sem_t sem_start;             // 启动信号量
    sem_t sem_complete;          // 完成信号量
    int data_ready;              // 数据就绪标志
    int error_code;              // 错误代码
    char hostname[256];          // 主机名验证
    size_t data_size;            // 数据字节数
    size_t tensor_dim;           // 张量维度
    unsigned char cuda_handle[CUDA_IPC_HANDLE_SIZE]; // CUDA IPC句柄存储
};

// 验证CUDA IPC句柄大小（编译期检查）
static_assert(sizeof(cudaIpcMemHandle_t) <= CUDA_IPC_HANDLE_SIZE, 
              "CUDA IPC handle size exceeds allocation");

void ipc_producer(int device_id, const char* shm_name, torch::Tensor tensor);
torch::Tensor ipc_consumer(int device_id, const char* shm_name);
void ipc_cleanup(const char* shm_name);