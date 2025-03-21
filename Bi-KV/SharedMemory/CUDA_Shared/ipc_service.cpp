#include "ipc_wrapper.h"
#include <pybind11/pybind11.h>
#include <chrono>
#include <string>

using namespace std::chrono;

// Producer static variables
static int producer_fd = -1;
static SharedControl* producer_ctrl = nullptr;
static void* producer_shared_mem = nullptr;
static char producer_shm_name[256];

// Consumer static variables
static int consumer_fd = -1;
static SharedControl* consumer_ctrl = nullptr;
static void* consumer_mapped_mem = nullptr;
static size_t BUFFER_SIZE=0;

void producer_init(int device_id, const char* shm_name, size_t buffer_size) {
    shm_unlink(shm_name);  // 新增代码
    BUFFER_SIZE=buffer_size;
    cudaSetDevice(device_id);
    strncpy(producer_shm_name, shm_name, sizeof(producer_shm_name) - 1);
    producer_shm_name[sizeof(producer_shm_name) - 1] = '\0';

    producer_fd = shm_open(shm_name, O_CREAT | O_RDWR | O_EXCL, 0666);
    if (producer_fd == -1) throw std::runtime_error("producer_init shm_open failed: " + std::string(strerror(errno)));

    if (ftruncate(producer_fd, sizeof(SharedControl)) == -1) {
        close(producer_fd);
        throw std::runtime_error("producer_init ftruncate failed: " + std::string(strerror(errno)));
    }

    producer_ctrl = static_cast<SharedControl*>(mmap(nullptr, sizeof(SharedControl), PROT_READ | PROT_WRITE, MAP_SHARED, producer_fd, 0));
    if (producer_ctrl == MAP_FAILED) {
        close(producer_fd);
        throw std::runtime_error("producer_init mmap failed: " + std::string(strerror(errno)));
    }

    memset(producer_ctrl, 0, sizeof(SharedControl));
    sem_init(&producer_ctrl->sem_start, 1, 0);
    sem_init(&producer_ctrl->sem_complete, 1, 0);
    gethostname(producer_ctrl->hostname, sizeof(producer_ctrl->hostname));
    producer_ctrl->current_offset = 0;
    producer_ctrl->last_valid_offset = 0;
    producer_ctrl->device_id = device_id; // 新增字段保存设备ID
    cudaError_t status = cudaMalloc(&producer_shared_mem, buffer_size);
    if (status != cudaSuccess) {
        munmap(producer_ctrl, sizeof(SharedControl));
        close(producer_fd);
        shm_unlink(shm_name);
        throw std::runtime_error("producer_init cudaMalloc failed: " + std::string(cudaGetErrorString(status)));
    }

    cudaIpcMemHandle_t handle;
    status = cudaIpcGetMemHandle(&handle, producer_shared_mem);
    if (status != cudaSuccess) {
        cudaFree(producer_shared_mem);
        munmap(producer_ctrl, sizeof(SharedControl));
        close(producer_fd);
        shm_unlink(shm_name);
        throw std::runtime_error("producer_init cudaIpcGetMemHandle failed: " + std::string(cudaGetErrorString(status)));
    }

    memcpy(producer_ctrl->cuda_handle, &handle, sizeof(handle));
}

void producer_send(torch::Tensor tensor) {
    TORCH_CHECK(tensor.device().is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(tensor.dtype() == torch::kFloat16, "Tensor must be float16");
    size_t data_size = tensor.nbytes();
    // 获取当前写入位置
    if (producer_ctrl->current_offset + data_size > BUFFER_SIZE) {
        producer_ctrl->current_offset = 0;  // 重置偏移量或抛出异常
    }
    void* write_ptr = static_cast<char*>(producer_shared_mem) + producer_ctrl->current_offset;
    cudaMemcpy(write_ptr, tensor.data_ptr(), data_size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();  // 确保拷贝完成
    producer_ctrl->data_size = data_size;
    producer_ctrl->tensor_dim = tensor.dim();
    producer_ctrl->last_valid_offset = producer_ctrl->current_offset;  // 记录有效数据起始位置
    producer_ctrl->current_offset += data_size;  // 移动偏移量
    // 记录形状信息
    producer_ctrl->tensor_dim = tensor.dim();
    for(int i=0; i<tensor.dim(); ++i){
        producer_ctrl->tensor_shape[i] = tensor.size(i);
    }
    

    sem_post(&producer_ctrl->sem_start);
    sem_wait(&producer_ctrl->sem_complete);
}

void producer_cleanup() {
    if (producer_ctrl) {
        sem_destroy(&producer_ctrl->sem_start);
        sem_destroy(&producer_ctrl->sem_complete);
        munmap(producer_ctrl, sizeof(SharedControl));
        producer_ctrl = nullptr;
    }
    if (producer_fd != -1) {
        close(producer_fd);
        shm_unlink(producer_shm_name);
        producer_fd = -1;
    }
    if (producer_shared_mem) {
        cudaFree(producer_shared_mem);
        producer_shared_mem = nullptr;
    }
}

void consumer_init(int device_id, const char* shm_name) {
    cudaSetDevice(device_id);
    consumer_fd = shm_open(shm_name, O_RDWR, 0666);
    if (consumer_fd == -1) throw std::runtime_error("consumer_init shm_open failed: " + std::string(strerror(errno)));

    consumer_ctrl = static_cast<SharedControl*>(mmap(nullptr, sizeof(SharedControl), PROT_READ | PROT_WRITE, MAP_SHARED, consumer_fd, 0));
    if (consumer_ctrl == MAP_FAILED) {
        close(consumer_fd);
        throw std::runtime_error("consumer_init mmap failed: " + std::string(strerror(errno)));
    }

    char current_host[256];
    gethostname(current_host, sizeof(current_host));
    if (strncmp(current_host, consumer_ctrl->hostname, sizeof(current_host)) != 0) {
        munmap(consumer_ctrl, sizeof(SharedControl));
        close(consumer_fd);
        throw std::runtime_error("Processes must run on the same host");
    }

    cudaIpcMemHandle_t handle;
    memcpy(&handle, consumer_ctrl->cuda_handle, sizeof(handle));

    cudaError_t status = cudaIpcOpenMemHandle(&consumer_mapped_mem, handle, cudaIpcMemLazyEnablePeerAccess);
    if (status != cudaSuccess) {
        munmap(consumer_ctrl, sizeof(SharedControl));
        close(consumer_fd);
        throw std::runtime_error("consumer_init cudaIpcOpenMemHandle failed: " + std::string(cudaGetErrorString(status)));
    }
}

torch::Tensor consumer_receive() {
    sem_wait(&consumer_ctrl->sem_start);
    int device_id=consumer_ctrl->device_id ; // 新增字段保存设备ID ;
     // 计算数据读取位置
    void* read_ptr = static_cast<char*>(consumer_mapped_mem) + consumer_ctrl->last_valid_offset;
    // 构建形状数组
    std::vector<int64_t> shape;
    for(int i=0; i<consumer_ctrl->tensor_dim; ++i){
        shape.push_back(consumer_ctrl->tensor_shape[i]);
    }
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(torch::kCUDA, device_id)
        .requires_grad(false);
    //std::vector<int64_t> shape(consumer_ctrl->tensor_dim, consumer_ctrl->data_size / sizeof(torch::Half));
    // 创建张量视图并克隆
    torch::Tensor result = torch::from_blob(
        read_ptr, 
        shape,  // 使用正确形状
        options
    ).clone();
    sem_post(&consumer_ctrl->sem_complete);
    return result;
}

void consumer_cleanup() {
    if (consumer_mapped_mem) {
        cudaIpcCloseMemHandle(consumer_mapped_mem);
        consumer_mapped_mem = nullptr;
    }
    if (consumer_ctrl) {
        munmap(consumer_ctrl, sizeof(SharedControl));
        consumer_ctrl = nullptr;
    }
    if (consumer_fd != -1) {
        close(consumer_fd);
        consumer_fd = -1;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("producer_init", &producer_init, "Init producer");
    m.def("producer_send", &producer_send, "Send data");
    m.def("producer_cleanup", &producer_cleanup, "Cleanup producer");
    m.def("consumer_init", &consumer_init, "Init consumer");
    m.def("consumer_receive", &consumer_receive, "Receive data");
    m.def("consumer_cleanup", &consumer_cleanup, "Cleanup consumer");
}