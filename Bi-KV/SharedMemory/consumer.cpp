#include "shared_control.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <cuda_runtime.h>

int main() {

     // === 关键代码：指定设备0 ===
     cudaError_t err = cudaSetDevice(4); // 使用第一个GPU
     if (err != cudaSuccess) {
         std::cerr << "Failed to set device 0: " 
                   << cudaGetErrorString(err) << std::endl;
         exit(1);
     }
    // ================= 获取GPU IPC句柄 =================
    cudaIpcMemHandle_t cuda_handle; 
    // (实际应从共享内存获取，此处简化为获取方式)
    
    // ================= 映射GPU内存 =================
    float* d_mapped;
    cudaIpcOpenMemHandle((void**)&d_mapped, cuda_handle, 
                        cudaIpcMemLazyEnablePeerAccess);

    // ================= 访问CPU控制内存 =================
    const char* shm_name = "/ctrl_shm";
    int fd = shm_open(shm_name, O_RDWR, 0666);
    SharedControl* ctrl = (SharedControl*)mmap(NULL, sizeof(SharedControl),
                                             PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

    // ================= 启动处理流程 =================
    if(ctrl->data_ready == 1) {
        sem_post(&ctrl->sem_start); // 通知生产者开始
        printf("Consumer: Starting processing...\n");

        // 等待数据完成
        sem_wait(&ctrl->sem_complete);
        
        // 验证结果
        if(ctrl->error_code == 0) {
            // 从d_mapped读取处理结果
            float result[1024];
            cudaMemcpy(result, d_mapped, 1024*sizeof(float), cudaMemcpyDeviceToHost);
            printf("Consumer: Data processed successfully\n");
        } else {
            printf("Error occurred: %d\n", ctrl->error_code);
        }
    }

    // 清理资源
    cudaIpcCloseMemHandle(d_mapped);
    return 0;
}