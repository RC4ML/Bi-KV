#include "ipc_wrapper.h"
#include <pybind11/pybind11.h>

void ipc_producer(int device_id, const char* shm_name, torch::Tensor tensor) {
    // 设置当前设备
    cudaSetDevice(device_id);
    TORCH_CHECK(tensor.device().is_cuda(), "Tensor must be on CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

    // 创建/打开共享内存
    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
    }

    // 调整共享内存大小
    if (ftruncate(fd, sizeof(SharedControl)) == -1) {
        close(fd);
        throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
    }

    // 内存映射
    SharedControl* ctrl = static_cast<SharedControl*>(
        mmap(nullptr, sizeof(SharedControl), 
             PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (ctrl == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
    }

    // 初始化控制结构
    memset(ctrl, 0, sizeof(SharedControl));
    sem_init(&ctrl->sem_start, 1, 0);
    sem_init(&ctrl->sem_complete, 1, 0);
    ctrl->data_ready = 1;
    ctrl->data_size = tensor.nbytes();
    ctrl->tensor_dim = tensor.dim();
    gethostname(ctrl->hostname, sizeof(ctrl->hostname));

    // 获取CUDA IPC句柄
    cudaIpcMemHandle_t handle;
    cudaError_t cudaStatus = cudaIpcGetMemHandle(&handle, tensor.data_ptr<float>());
    if (cudaStatus != cudaSuccess) {
        munmap(ctrl, sizeof(SharedControl));
        close(fd);
        shm_unlink(shm_name);
        throw std::runtime_error("cudaIpcGetMemHandle failed: " 
                               + std::string(cudaGetErrorString(cudaStatus)));
    }

    // 存储二进制句柄
    memcpy(ctrl->cuda_handle, &handle, sizeof(cudaIpcMemHandle_t));

    // 等待消费者启动
    sem_wait(&ctrl->sem_start);  // 初始为0，阻塞等待

    // 处理完成后通知
    sem_post(&ctrl->sem_complete); // 值从0→1，唤醒消费者
    
    // 清理资源
    sem_destroy(&ctrl->sem_start);
    sem_destroy(&ctrl->sem_complete);
    munmap(ctrl, sizeof(SharedControl));
    close(fd);
    //std::cout <<"producer finfish"<<std::endl ;
    return ;
}

torch::Tensor ipc_consumer(int device_id, const char* shm_name) {
    // 设置当前设备
    cudaSetDevice(device_id);
    
    // 打开共享内存
    int fd = shm_open(shm_name, O_RDWR, 0666);
    if (fd == -1) {
        throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
    }

    // 内存映射
    SharedControl* ctrl = static_cast<SharedControl*>(
        mmap(nullptr, sizeof(SharedControl),
             PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (ctrl == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
    }

    // 主机验证
    char current_host[256];
    gethostname(current_host, sizeof(current_host));
    if (strncmp(current_host, ctrl->hostname, sizeof(current_host)) != 0) {
        munmap(ctrl, sizeof(SharedControl));
        close(fd);
        throw std::runtime_error("Processes must run on the same host");
    }

    // 提取CUDA句柄
    cudaIpcMemHandle_t handle;
    memcpy(&handle, ctrl->cuda_handle, sizeof(cudaIpcMemHandle_t));

    // 映射设备内存
    void* d_mapped = nullptr;
    cudaError_t cudaStatus = cudaIpcOpenMemHandle(
        &d_mapped, handle, cudaIpcMemLazyEnablePeerAccess);
    if (cudaStatus != cudaSuccess) {
        munmap(ctrl, sizeof(SharedControl));
        close(fd);
        throw std::runtime_error("cudaIpcOpenMemHandle failed: " 
                               + std::string(cudaGetErrorString(cudaStatus)));
    }

    // 创建Tensor并克隆
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA, device_id)
        .requires_grad(false);
    torch::Tensor result = torch::from_blob(
        d_mapped, 
        std::vector<int64_t>(ctrl->tensor_dim, ctrl->data_size / sizeof(float)),
        options
    ).clone();

    // 通知生产者完成
    sem_post(&ctrl->sem_start);
    // 等待处理完成
    sem_wait(&ctrl->sem_complete); // 初始为0，阻塞等待
    // 清理资源
    cudaIpcCloseMemHandle(d_mapped);
    munmap(ctrl, sizeof(SharedControl));
    close(fd);
    //std::cout <<"consumer finfish"<<std::endl ;
    return result;
}

void ipc_cleanup(const char* shm_name) {
    if (shm_unlink(shm_name) == -1 && errno != ENOENT) {
        throw std::runtime_error("shm_unlink failed: " 
                               + std::string(strerror(errno)));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("producer", &ipc_producer, "IPC Producer");
    m.def("consumer", &ipc_consumer, "IPC Consumer");
    m.def("cleanup", &ipc_cleanup, "Cleanup IPC resources");
}