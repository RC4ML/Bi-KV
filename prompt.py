当cache_worker=infer_worker时使用共享内存，目前最大的延时为“time0 = time.time()
        expected_numel = total_token_num * np.prod(token_shape)
        assert send_tensor.numel() == expected_numel, \
            f"形状不匹配: {send_tensor.shape} vs 预期{token_shape}"
        indices = torch.empty(total_token_num, dtype=torch.long)
        offset = 0
        circle_counter = 0
        for idx, page_list in enumerate(cache_pages_list):
            id_token_pair = id_token_pair_list[idx]
            item_token_num = id_token_pair[1]
            for page_idx, page in enumerate(page_list):
                start = page * self.page_size
                circle_counter += 1
                if page_idx == len(page_list) - 1:
                    size = (item_token_num % self.page_size) if (item_token_num % self.page_size != 0) else self.page_size
                    indices[offset:offset + size] = torch.arange(start, start + size)
                    offset += size
                else:
                    indices[offset:offset + self.page_size] = torch.arange(start, start + self.page_size)
                    offset += self.page_size

        send_tensor[:] = self.cache_data[indices]
        time1 = time.time()”  从page中整理出连续tensor,请写出直接使用gpu kernel并行从cache的page写入worker的page的代码（GPU到GPU），并详细解释
for task_infer_worker in combined_task_info:
            combined_task_list = combined_task_info[task_infer_worker]
            for task_info in combined_task_list.values():
                task_type = task_info['task_type']
                infer_worker = task_info['infer_worker']
                TokenNum=task_info['token_num']
                infer_worker_port = 2*infer_worker + WORKER_offset
                infer_worker_addr = f"localhost:{self.master_port+infer_worker_port}"
                if task_type == SIGNAL_SEND:
                    if DEBUG:
                        print(f"[KVCache.receive_task_info_batch][RANK {self.rank}]{task_info}")
                        print(f"[KVCache {self.rank}] 执行Send请求 - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
                    combined_task_info_pb = self._task_info_json_to_pb(task_info)
                    start_time=time.time()
                    if cache_worker== infer_worker:
                        shared_start=time.time()
                        self.shared_data_batch(task_info)  # 先写入CUDA共享内存，再call RPC,避免worker等待cache写入共享内存
                        shared_end=time.time()
                        print(f"self.shared_data_batch{shared_end-shared_start}")
                    with grpc.insecure_channel(infer_worker_addr) as channel:
                        stub = TaskInfo_pb2_grpc.InferWorkerServiceStub(channel)
                        remote_recv = stub.RecvKVCacheData.future(combined_task_info_pb)
                        if cache_worker== infer_worker: # on the same device,use CUDA shared Memory
                            # self.shared_data_batch(task_info)
                            remote_recv.result()
                            time_share2=time.time()
                            with open(f'e2e_log_rank_shared{self.rank}.txt', 'a+') as f:
                                f.write(f"e2e shared once time: {time_share2-start_time}s,token_num:{TokenNum}, throughput: {((TokenNum*8*28*128)/(time_share2-start_time)/(1e9))} GB/s\n")
                            print(f"e2e shared once time: {time_share2-start_time}s,token_num:{TokenNum}, throughput: {((TokenNum*8*28*128)/(time_share2-start_time)/(1e9))} GB/s")
                            # time_send1=time.time()
                            # self.send_data_batch(task_info)
                            # remote_recv.result()
                            # time_send2=time.time()
                            # with open(f'kvcache_log_rank_send{self.rank}.txt', 'a') as f:
                            #     f.write(f"send once time: {time_send2-time_send1}s,token_num:{TokenNum}, throughput: {((TokenNum*8*28*128)/(time_send2-time_send1)/(1e9))} GB/s\n")
                            # print(f"send once time: {time_send2-time_send1}s,token_num:{token_num}, throughput: {((token_num*8*28*128)/(time_send2-time_send1)/(1e9))} GB/s")
                        else:
                            #time_send1=time.time()
                            self.send_data_batch(task_info)
                            remote_recv.result()
                            time_send2=time.time()
                            with open(f'e2e_log_rank_send{self.rank}.txt', 'a+') as f:
                                f.write(f"e2e send once time: {time_send2-start_time}s,token_num:{TokenNum}, throughput: {((TokenNum*8*28*128)/(time_send2-start_time)/(1e9))} GB/s\n")
                            print(f"e2e send once time: {time_send2-start_time}s,token_num:{TokenNum}, throughput: {((TokenNum*8*28*128)/(time_send2-start_time)/(1e9))} GB/s")
                    # print(f"[KVCache][RANK {self.rank}] 执行Send请求完成 - cacheRank {2*cache_worker+3} -> workerRank {2*infer_worker+2}")
                    self.send_counter += 1
def shared_data_batch(self, combined_task_info: Dict):
        """使用CUDA共享内存传输数据"""
        dst_rank = 2*combined_task_info['infer_worker'] + WORKER_offset
        token_num = combined_task_info['token_num']
        id_token_pair_list = combined_task_info['id_token_pair']
        cache_pages_list = combined_task_info['cache_pages_list']
        if DEBUG:
            print(f"[KVCache][Rank {self.rank}] 开始发送数据到 Rank {dst_rank}, 长度={token_num}")
        # 计算总 token 数
        total_token_num = sum(id_token_pair[1] for id_token_pair in id_token_pair_list)
        time0 = time.time()
        # 一次性分配大 tensor
        send_tensor = torch.empty(
            (total_token_num,) + token_shape,
            dtype=torch.float16,
            device='cuda'
        )
        time1 = time.time()
        print(f"一次性分配大 tensor{time1-time0}")
         # 添加形状校验
        time0 = time.time()
        expected_numel = total_token_num * np.prod(token_shape)
        assert send_tensor.numel() == expected_numel, \
            f"形状不匹配: {send_tensor.shape} vs 预期{token_shape}"
        indices = torch.empty(total_token_num, dtype=torch.long)
        offset = 0
        circle_counter = 0
        for idx, page_list in enumerate(cache_pages_list):
            id_token_pair = id_token_pair_list[idx]
            item_token_num = id_token_pair[1]
            for page_idx, page in enumerate(page_list):
                start = page * self.page_size
                circle_counter += 1
                if page_idx == len(page_list) - 1:
                    size = (item_token_num % self.page_size) if (item_token_num % self.page_size != 0) else self.page_size
                    indices[offset:offset + size] = torch.arange(start, start + size)
                    offset += size
                else:
                    indices[offset:offset + self.page_size] = torch.arange(start, start + self.page_size)
                    offset += self.page_size

        send_tensor[:] = self.cache_data[indices]
        time1 = time.time()
        print(f"提取pages{time1-time0}")
        if DEBUG:
            print(f"[KVCache]共享内存[Rank {self.rank}] send_tensor shape: {send_tensor.size()} token num: {token_num}")
        time0 = time.time()
        ipc_service.producer_send(send_tensor)
        time1 = time.time()
        print(f"ipc_service.producer_send{time1-time0}")
        if DEBUG:
            print(f"内部shared once time: {time1-time0}s, total_token_num: {total_token_num},throughput: {((token_num*8*28*128)/(time1-time0)/(1e9))} GB/s")
        if DEBUG:
            print(f"[KVCache]共享内存[Rank {self.rank}] 完成发送数据到 Rank {dst_rank}, 长度={token_num}")
def receive_kvcache_data_batch(self, combined_task_info):
        cache_worker = combined_task_info.cache_worker
        src_rank = 2*cache_worker + KVCACHE_offset
        token_num = combined_task_info.token_num
        if self.worker_index == cache_worker:
            print(f"[Worker][RANK {self.rank}] start get shared memory")
            #从共享内存接收CUDA张量
            start_read=time.time()
            cuda_tensor=ipc_service.consumer_receive()
            end_read=time.time()      
            #将张量复制到CPU
            recv_tensor = cuda_tensor.cpu()
            print(f"shared{recv_tensor.size()}")
            del cuda_tensor
            torch.cuda.empty_cache()  # 可选但建议添加
            #计算总数据量（字节）
            total_bytes = recv_tensor.numel() * recv_tensor.element_size()  # 正确计算总字节
            time_diff = end_read - start_read
            throughput = total_bytes / time_diff / 1e9  # 转换为GB/s
            with open(f'shared_log_rank{self.rank}.txt', 'a+') as f:
                f.write(f"[ipc_service.consumer_receive]shared once time: {time_diff}s, torch.size{recv_tensor.size()},total_bytes:{total_bytes/(1024**2)}MB, "
                    f"throughput: {throughput} GB/s\n")
        else: 
            start_recv=time.time()
            recv_tensor = torch.empty(
            (token_num,) + token_shape, 
            dtype=torch.float16
            )
            dist.recv(tensor=recv_tensor, src=src_rank)
            end_recv=time.time()
            #计算总数据量（字节）
            total_bytes = recv_tensor.numel() * recv_tensor.element_size()  # 正确计算总字节
            time_diff = end_recv - start_recv
            throughput = total_bytes / time_diff / 1e9  # 转换为GB/s
            with open(f'send_log_rank{self.rank}.txt', 'a+') as f:
                f.write(f"[dist.recv]send once time: {time_diff}s, torch.size{recv_tensor.size()},total_bytes:{total_bytes/(1024**2)}MB, "
                f"throughput: {throughput} GB/s\n")
            print(f"send{recv_tensor.size()}")
        # 计算总大小并预分配索引 tensor
        total_size = sum(id_pair.token_num for id_pair in combined_task_info.id_token_pair)
        indices = torch.empty(total_size, dtype=torch.long)
        buffer_indices = torch.empty(total_size, dtype=torch.long)

        # 第一步：收集索引
        start_pos = 0
        buffer_pos = 0
        for id_pair in combined_task_info.id_token_pair:
            item_id = id_pair.id
            offset = id_pair.token_num
            
            # 管理 page
            if item_id not in self.page_manager.get_loaded_lists():
                page_set, _ = self.page_manager.load_item(item_id, offset)
            else:
                page_set = self.page_manager.access_item(item_id)
            
            # 生成索引
            item_size = offset
            indices[start_pos:start_pos + item_size] = torch.arange(start_pos, start_pos + item_size)
            
            # 生成 buffer 对应的索引
            for idx, page in enumerate(page_set):
                if idx == len(page_set) - 1:
                    size = offset % self.page_size if offset % self.page_size != 0 else self.page_size
                    buffer_indices[buffer_pos:buffer_pos + size] = torch.arange(
                        page * self.page_size, 
                        page * self.page_size + size
                    )
                    buffer_pos += size
                else:
                    buffer_indices[buffer_pos:buffer_pos + self.page_size] = torch.arange(
                        page * self.page_size, 
                        (page + 1) * self.page_size
                    )
                    buffer_pos += self.page_size
            
            start_pos += offset

        # 第二步：一次性提取数据并写入 buffer
        self.compute_buffer[buffer_indices] = recv_tensor[indices]
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
    for (int i = 0; i < consumer_ctrl->tensor_dim; ++i) {
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("producer_init", &producer_init, "Init producer");
    m.def("producer_send", &producer_send, "Send data");
    m.def("producer_cleanup", &producer_cleanup, "Cleanup producer");
    m.def("consumer_init", &consumer_init, "Init consumer");
    m.def("consumer_receive", &consumer_receive, "Receive data");
    m.def("consumer_cleanup", &consumer_cleanup, "Cleanup consumer");
}
from config import *
import time
import threading

eviction_lock = threading.Lock()
# allocate_lock = threading.Lock()

class PageManager:
    def __init__(self, cache_size, page_size, pm_id = 0):
        self.pm_id = pm_id
        # self.buffer = cache_size*['']
        self.page_size = page_size
        self.num_pages = cache_size // page_size
        # set是不是不合适，换成队列？
        self.free_pages = set(range(self.num_pages))
        self.page_table = {}  # 键为item ID，值为{'pages': set, 'last_accessed': int}
        self.current_time = 0  # 模拟时间戳，用于LRU
        print(f"初始空闲页数:{len(self.free_pages)}")

    def load_item(self, item_id, list_length):
        # 如果id已经存在则直接返回
        if item_id in self.page_table:
            return self.access_item(item_id), []
        """加载列表，必要时换出旧列表"""
        required_pages = (list_length + self.page_size - 1) // self.page_size
        if required_pages > self.num_pages:
            raise ValueError("列表过大，无法存入缓冲区")

        freed_ids = []

        # eviction_lock.acquire()
        # 检查是否需要换出
        if len(self.free_pages) < required_pages:
            freed_ids = self._perform_eviction(required_pages)

        # 分配新页
        allocated_pages = self._allocate_pages(required_pages)
        # 目前的情况看不加锁也没有问题
        # eviction_lock.release()
        self.page_table[item_id] = {
            'pages': allocated_pages,
            'last_accessed': self.current_time,
            'protected': 0
        }
        # for i in allocated_pages:
        #     for j in range(self._page_to_location(i), self._page_to_location(i)+self.page_size):
        #         self.buffer[j] = item_id
        self.current_time += 1
        return allocated_pages,freed_ids

    def access_item(self, item_id) -> set:
        """访问列表，更新最近使用时间，返回页号"""
        if item_id not in self.page_table:
            raise KeyError(f"列表 {item_id} 未加载")
        self.page_table[item_id]['last_accessed'] = self.current_time
        self.current_time += 1
        return self.page_table[item_id]['pages']

    def _perform_eviction(self, required_pages):
        """执行换出操作直到有足够空间"""
        # protected_items = sum(len(info['pages']) for info in self.page_table.values() if info['protected'] > 0)
        # paged_items = sum(len(info['pages']) for info in self.page_table.values())
        # print(f"触发换出操作，当前空闲页数:{len(self.free_pages)} 保护数量:{protected_items} 占用数量：{paged_items} 要求页数:{required_pages}")
        # 按LRU排序现有列表
        lru_entries = sorted(
            self.page_table.items(),
            key=lambda item: item[1]['last_accessed'],
        )

        freed_ids = []
        for item_id, info in lru_entries:
            # 如果保护则继续
            if self.page_table[item_id]['protected']!=0:
                continue
            # 换出该列表
            del self.page_table[item_id]
            self.free_pages.update(info['pages'])
            freed_ids.append(item_id)
            if DEBUG:
                print(f"[[{time.time()}]] 换出列表 {item_id}，释放页数 {len(info['pages'])} 空闲页数:{len(self.free_pages)}")
            if len(self.free_pages) >= required_pages:
                break

        if len(self.free_pages) < required_pages:
            # protected_items = sum(len(info['pages']) for info in self.page_table.values() if info['protected'] > 0)
            # print(f"[[{time.time()}]] 无法换出足够页面，当前页面:{list(self.page_table.keys())} 占用数量:{protected_items}")
            raise RuntimeError(f"Unable to free enough pages, even if all items are swapped out") 
        # 返回换出的页号
        return freed_ids

    def set_protected(self, item_id):
        """保护列表不被换出"""
        if item_id not in self.page_table:
            raise KeyError(f"列表 {item_id} 未加载")
        self.page_table[item_id]['protected'] += 1
        # print(f"[[{time.time()}]] 保护item {item_id} {self.page_table[item_id]['protected']}次")

    def remove_protected(self, item_id):
        """取消保护"""
        if item_id not in self.page_table:
            raise KeyError(f"列表 {item_id} 未加载")
        self.page_table[item_id]['protected'] -= 1
        # print(f"[[{time.time()}]] 取消保护item {item_id} {self.page_table[item_id]['protected']}次")

    def _allocate_pages(self, n):
        """分配指定数量的页（无需连续）"""
        if len(self.free_pages) < n:
            raise RuntimeError("内部错误：分配时页面不足")
        allocated = set()
        for _ in range(n):
            # 取第一个可用页（实现可优化）
            page = next(iter(self.free_pages))
            allocated.add(page)
            self.free_pages.remove(page)
        return allocated
    
    def _page_to_location(self, page):
        """将页号转换为实际位置"""
        return page * self.page_size

    def get_loaded_lists(self):
        """返回当前加载的列表ID"""
        return list(self.page_table.keys())

class MultiPageManager:
    '''多个PageManager组成的KV缓存'''
    def __init__(self, cache_size, page_size, kvcahe_num):
        self.kvcahe_num = kvcahe_num
        self.page_managers = [PageManager(cache_size, page_size, pm_id=i) for i in range(kvcahe_num)]
        self.buffer_size = cache_size
        self.page_size = page_size
        self.num_pages = cache_size // page_size
        # 记录每个pm的缓存列表
        self.cached_ids = [set() for _ in range(kvcahe_num)]

    def load_item(self, item_id, list_length):
        """加载列表，必要时换出旧列表"""
        # NOTE 需要注意的是，应该先查询在不在缓存中，如果在缓存中则直接返回页号
        # 计算最大剩余页数
        max_page_num = max(len(pm.free_pages) for pm in self.page_managers)
        if max_page_num>(self.num_pages/10):
            # 选择剩余页数最多的pm
            target_pm = max(self.page_managers, key=lambda x: len(x.free_pages))
        else:
            # 随机选择一个pm 防止负载不均衡
            target_pm = random.choice(self.page_managers)
        target_pm_id = target_pm.pm_id
        allocated_pages, freed_ids = target_pm.load_item(item_id, list_length)
        # 记录已缓存的列表
        self.cached_ids[target_pm_id].add(item_id)
        # 删除已换出的列表
        for freed_id in freed_ids:
            self.cached_ids[target_pm_id].remove(freed_id)
        return target_pm_id, allocated_pages

    def access_item(self, item_id):
        """访问列表，更新最近使用时间，返回页号，否则返回None"""
        for idx, cached_id in enumerate(self.cached_ids):
            if item_id in cached_id:
                return idx,self.page_managers[idx].access_item(item_id)
        return None, None

# 示例用法
if __name__ == "__main__":
    # 初始化：32元素缓冲区，每页4元素 → 8页
    pm = PageManager(cache_size=32, page_size=4)

    # 加载列表A（5元素 → 2页）
    print(pm.load_item("A", 5))  # 例如输出 {0, 1}
    print("当前加载的列表:", pm.get_loaded_lists())
    # print("当前缓冲区状态:", pm.buffer)


    # 加载列表B（7元素 → 2页）
    print(pm.load_item("B", 7))  # 例如输出 {2, 3}
    print("当前加载的列表:", pm.get_loaded_lists())
    # print("当前缓冲区状态:", pm.buffer)


    # 访问列表A更新LRU时间
    pm.access_item("A")

    # 加载列表C（10元素 → 3页，需换出B）
    print(pm.load_item("C", 20))  # 例如输出 {4, 5, 6}
    print("当前加载的列表:", pm.get_loaded_lists())  # 应包含A和C
    # print("当前缓冲区状态:", pm.buffer)


    # 加载列表C（10元素 → 3页，需换出B）
    print(pm.load_item("D", 30))  # 例如输出 {4, 5, 6}
    print("当前加载的列表:", pm.get_loaded_lists())  # 应包含A和C
    # print("当前缓冲区状态:", pm.buffer)

    #测试 MultiPageManager
    mpm = MultiPageManager(32, 4, 2)
    mpm.load_item("A", 5)
    mpm.load_item("B", 7)
    mpm.load_item("C", 20)
    mpm.load_item("D", 30)
    print(mpm.access_item("A"))
    print(mpm.access_item("B"))
    print(mpm.access_item("C"))
    print(mpm.access_item("D"))
