//ipc_service.cpp
#include "ipc_wrapper.h"
#include <pybind11/pybind11.h>
#include <chrono>
#include <string>
#include <iostream>

// Producer static variables
static int producer_fd = -1;
static SharedControl* producer_ctrl = nullptr;
static void* producer_shared_mem = nullptr;

// Consumer static variables
static int consumer_fd = -1;
static SharedControl* consumer_ctrl = nullptr;
static void* consumer_shared_mem = nullptr;
static char consumer_shm_name[256];

// 修改后的 consumer_init，由Consumer创建共享内存和CUDA内存
void consumer_init(int device_id, const char* shm_name, size_t buffer_size) {
    cudaSetDevice(device_id);
    strncpy(consumer_shm_name, shm_name, sizeof(consumer_shm_name) - 1);
    consumer_shm_name[sizeof(consumer_shm_name) - 1] = '\0';

    // 确保共享内存不存在
    shm_unlink(shm_name);

    // 创建共享内存并设置大小
    consumer_fd = shm_open(shm_name, O_CREAT | O_RDWR | O_EXCL, 0666);
    if (consumer_fd == -1) throw std::runtime_error("consumer_init shm_open failed: " + std::string(strerror(errno)));

    if (ftruncate(consumer_fd, sizeof(SharedControl)) == -1) {
        close(consumer_fd);
        throw std::runtime_error("consumer_init ftruncate failed: " + std::string(strerror(errno)));
    }

    // 映射共享内存控制结构
    consumer_ctrl = static_cast<SharedControl*>(mmap(nullptr, sizeof(SharedControl), PROT_READ | PROT_WRITE, MAP_SHARED, consumer_fd, 0));
    if (consumer_ctrl == MAP_FAILED) {
        close(consumer_fd);
        throw std::runtime_error("consumer_init mmap failed: " + std::string(strerror(errno)));
    }

    // 初始化控制结构
    memset(consumer_ctrl, 0, sizeof(SharedControl));
    sem_init(&consumer_ctrl->sem_start, 1, 0);
    sem_init(&consumer_ctrl->sem_complete, 1, 0);
    gethostname(consumer_ctrl->hostname, sizeof(consumer_ctrl->hostname));
    consumer_ctrl->current_offset = 0;
    consumer_ctrl->last_valid_offset = 0;
    consumer_ctrl->device_id = device_id;
    consumer_ctrl->buffer_size = buffer_size;

    // 分配CUDA内存
    cudaError_t status = cudaMalloc(&consumer_shared_mem, buffer_size);
    if (status != cudaSuccess) {
        munmap(consumer_ctrl, sizeof(SharedControl));
        close(consumer_fd);
        shm_unlink(shm_name);
        throw std::runtime_error("consumer_init cudaMalloc failed: " + std::string(cudaGetErrorString(status)));
    }

    // 获取IPC句柄并存储到共享内存
    cudaIpcMemHandle_t handle;
    status = cudaIpcGetMemHandle(&handle, consumer_shared_mem);
    if (status != cudaSuccess) {
        cudaFree(consumer_shared_mem);
        munmap(consumer_ctrl, sizeof(SharedControl));
        close(consumer_fd);
        shm_unlink(shm_name);
        throw std::runtime_error("consumer_init cudaIpcGetMemHandle failed: " + std::string(cudaGetErrorString(status)));
    }
    memcpy(consumer_ctrl->cuda_handle, &handle, sizeof(handle));
}

// 修改后的 producer_init，连接到Consumer创建的共享内存
void producer_init(int device_id, const char* shm_name) {
    cudaSetDevice(device_id);

    // 打开共享内存
    producer_fd = shm_open(shm_name, O_RDWR, 0666);
    if (producer_fd == -1) throw std::runtime_error("producer_init shm_open failed: " + std::string(strerror(errno)));

    // 映射控制结构
    producer_ctrl = static_cast<SharedControl*>(mmap(nullptr, sizeof(SharedControl), PROT_READ | PROT_WRITE, MAP_SHARED, producer_fd, 0));
    if (producer_ctrl == MAP_FAILED) {
        close(producer_fd);
        throw std::runtime_error("producer_init mmap failed: " + std::string(strerror(errno)));
    }

    // 验证同一主机
    char current_host[256];
    gethostname(current_host, sizeof(current_host));
    if (strncmp(current_host, producer_ctrl->hostname, sizeof(current_host)) != 0) {
        munmap(producer_ctrl, sizeof(SharedControl));
        close(producer_fd);
        throw std::runtime_error("Processes must run on the same host");
    }

    // 通过IPC句柄打开CUDA内存
    cudaIpcMemHandle_t handle;
    memcpy(&handle, producer_ctrl->cuda_handle, sizeof(handle));

    cudaError_t status = cudaIpcOpenMemHandle(&producer_shared_mem, handle, cudaIpcMemLazyEnablePeerAccess);
    if (status != cudaSuccess) {
        munmap(producer_ctrl, sizeof(SharedControl));
        close(producer_fd);
        throw std::runtime_error("producer_init cudaIpcOpenMemHandle failed: " + std::string(cudaGetErrorString(status)));
    }
}

// Producer发送数据到共享内存
void producer_send(torch::Tensor tensor) {
    TORCH_CHECK(tensor.device().is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(tensor.dtype() == torch::kFloat16, "Tensor must be float16");
    size_t data_size = tensor.nbytes();

    // 检查缓冲区溢出
    if (producer_ctrl->current_offset + data_size > producer_ctrl->buffer_size) {
        producer_ctrl->current_offset = 0;
    }

    // 数据拷贝到共享内存
    void* write_ptr = static_cast<char*>(producer_shared_mem) + producer_ctrl->current_offset;
    cudaMemcpy(write_ptr, tensor.data_ptr(), data_size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    // 更新控制信息
    producer_ctrl->data_size = data_size;
    producer_ctrl->tensor_dim = tensor.dim();
    producer_ctrl->last_valid_offset = producer_ctrl->current_offset;
    producer_ctrl->current_offset += data_size;

    // 记录张量形状
    for (int i = 0; i < tensor.dim(); ++i) {
        producer_ctrl->tensor_shape[i] = tensor.size(i);
    }

    // 通知Consumer
    sem_post(&producer_ctrl->sem_start);
    //sem_wait(&producer_ctrl->sem_complete);
}

// Consumer接收数据
torch::Tensor consumer_receive() {
    sem_wait(&consumer_ctrl->sem_start);

    // 构造张量
    void* read_ptr = static_cast<char*>(consumer_shared_mem) + consumer_ctrl->last_valid_offset;
    std::vector<int64_t> shape;
    for (size_t i = 0; i < consumer_ctrl->tensor_dim; ++i) {
        shape.push_back(consumer_ctrl->tensor_shape[i]);
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(torch::kCUDA, consumer_ctrl->device_id)
        .requires_grad(false);

    torch::Tensor result = torch::from_blob(
        read_ptr,
        shape,
        options
    );
    //sem_post(&consumer_ctrl->sem_complete);
    return result;
}

// 清理函数
void producer_cleanup() {
    if (producer_shared_mem) {
        cudaIpcCloseMemHandle(producer_shared_mem);
        producer_shared_mem = nullptr;
    }
    if (producer_ctrl) {
        munmap(producer_ctrl, sizeof(SharedControl));
        producer_ctrl = nullptr;
    }
    if (producer_fd != -1) {
        close(producer_fd);
        producer_fd = -1;
    }
}

void consumer_cleanup() {
    if (consumer_shared_mem) {
        cudaFree(consumer_shared_mem);
        consumer_shared_mem = nullptr;
    }
    if (consumer_ctrl) {
        sem_destroy(&consumer_ctrl->sem_start);
        sem_destroy(&consumer_ctrl->sem_complete);
        munmap(consumer_ctrl, sizeof(SharedControl));
        consumer_ctrl = nullptr;
    }
    if (consumer_fd != -1) {
        close(consumer_fd);
        shm_unlink(consumer_shm_name);
        consumer_fd = -1;
    }
}

// 新增：直接页面复制函数
void producer_copy_pages(
    torch::Tensor cache_data,
    torch::Tensor src_offsets,
    torch::Tensor dest_offsets,
    int page_size,
    const TensorDims& dims 
) {
    cuda_producer_copy_pages(
        cache_data,
        src_offsets,
        dest_offsets,
        producer_ctrl,
        producer_shared_mem,
        page_size,
        dims
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<TensorDims>(m, "TensorDims")
        .def(pybind11::init<>())
        .def_readwrite("total_tokens", &TensorDims::total_tokens)
        .def_readwrite("head_size", &TensorDims::head_size)
        .def_readwrite("num_kv_heads", &TensorDims::num_kv_heads)
        .def_readwrite("num_layers", &TensorDims::num_layers)
        .def_readwrite("kv_pair", &TensorDims::kv_pair);
    m.def("producer_init", &producer_init, "Init producer");
    m.def("producer_send", &producer_send, "Send data");
    m.def("producer_cleanup", &producer_cleanup, "Cleanup producer");
    m.def("consumer_init", &consumer_init, "Init consumer");
    m.def("consumer_receive", &consumer_receive, "Receive data");
    m.def("consumer_cleanup", &consumer_cleanup, "Cleanup consumer");
    m.def("producer_copy_pages", &producer_copy_pages, "Direct page copy to shared memory");
    m.def("cuda_producer_copy_pages", &cuda_producer_copy_pages, "cuda Direct page copy to shared memory");
}