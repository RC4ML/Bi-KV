#include "shared_control.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>  
#include <cuda_runtime.h>

int main() {
    // === 关键代码：指定同一设备 ===
    cudaSetDevice(4); // 必须与生产者设备一致
    // ================= GPU数据初始化 =================
    float* d_data;
    cudaMalloc(&d_data, 1024*sizeof(float)); // 分配GPU内存
    cudaMemset(d_data, 0, 1024*sizeof(float)); // 初始化数据

    // ================= 创建CUDA IPC句柄 =================
    cudaIpcMemHandle_t cuda_handle;
    cudaIpcGetMemHandle(&cuda_handle, d_data); // 获取IPC句柄

    // ================= CPU控制共享内存 =================
    const char* shm_name = "/ctrl_shm";
    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(SharedControl));
    SharedControl* ctrl = (SharedControl*)mmap(NULL, sizeof(SharedControl), 
                                             PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    
    // 初始化信号量和控制标志
    sem_init(&ctrl->sem_start, 1, 0);
    sem_init(&ctrl->sem_complete, 1, 0);
    ctrl->data_ready = 1; // 标记数据已就绪
    ctrl->error_code = 0;

    // ================= 等待消费者启动 =================
    printf("Producer: Waiting for consumer...\n");
    sem_wait(&ctrl->sem_start); // 等待开始信号

    // ================= 数据处理阶段 =================
    // （此处可添加CUDA内核调用处理数据）
    
    // 通知完成
    sem_post(&ctrl->sem_complete);
    printf("Producer: Processing completed\n");

    // 清理资源
    cudaFree(d_data);
    shm_unlink(shm_name);
    return 0;
}