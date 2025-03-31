#ifndef RDMA_ONESIDED_ENDPOINT_H
#define RDMA_ONESIDED_ENDPOINT_H

#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
#include <string>
#include <map>
#include <vector>
#include <cstring>
#include <iostream>
#include <torch/extension.h>
#include <sys/mman.h>
#include <arpa/inet.h>
#include <cstdlib>
#include <errno.h>
#include <cuda_runtime.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

// 在头文件中添加结构体定义（需严格对齐）
#pragma pack(push, 1)
struct Metadata {
    uint64_t addr;      // 内存地址
    uint32_t rkey;      // 远程密钥
    size_t size;        // 内存大小（注意平台兼容性）
    char type[4];       // "cpu" 或 "gpu"（固定4字节）
    bool allocated;     // 是否已分配（1字节）
    int rank_id;        // 客户端的 rank ID（4字节）
};
#pragma pack(pop)  // 总大小: 8+4+8+4+1+4=29字节
/**
 * @brief Stores remote memory region information
 */
struct RemoteMemoryInfo {
    uint64_t addr;    // Remote memory address
    uint32_t rkey;    // Remote memory access key
    size_t size;      // Memory region size
    std::string type; // "cpu" or "gpu"
    bool allocated;   // Whether this memory is allocated
};

/**
 * @brief RDMAOneSidedEndpoint
 * Encapsulates RDMA connection setup, one-sided read/write operations, and metadata exchange.
 *
 * Features:
 * - Server allocates independent Read and Write MRs for each client.
 * - Supports RDMA read/write from/to client CPU/GPU memory to server CPU/GPU memory.
 * - Internal memory allocation for CPU and GPU; users specify sizes and allocation flags.
 * - Each client gets at most one CPU and one GPU MR.
 * - Buffers accessible as PyTorch tensors after initialization.
 */
class RDMAOneSidedEndpoint {
public:
    RDMAOneSidedEndpoint(const std::string& ip, 
                         const std::string& port, 
                         const std::string& mode)
        : ip_(ip), port_(port), mode_(mode), 
          ec_(nullptr), cm_id_(nullptr), pd_(nullptr), 
          local_cpu_mr_(nullptr), local_gpu_mr_(nullptr),
          local_cpu_buffer_(nullptr), local_gpu_buffer_(nullptr),
          local_cpu_buffer_size_(0), local_gpu_buffer_size_(0),
          local_cpu_allocated_(false), local_gpu_allocated_(false),
          use_hugepage_(false), rank_id_(-1) {
        // Environment checks
        if (!check_rdma_device()) {
            throw std::runtime_error("No RDMA device found.");
        }
        if (!check_network_config()) {
            throw std::runtime_error("Network configuration error.");
        }
    }

    ~RDMAOneSidedEndpoint() {
        cleanup();
    }

    int run_server(int max_clients, size_t local_cpu_size, size_t local_gpu_size, bool hugepage = false) {
        // 创建事件通道和 cm_id
        if (create_event_channel() != 0) return -1;
        if (create_id() != 0) return -1;

        // 绑定地址并监听
        sockaddr_in addr = create_addr(INADDR_ANY);
        if (bind_and_listen(&addr) != 0) return -1;

        // 接受客户端连接
        int clients = 0;
        // local_cpu_buffer_size_ = local_cpu_size;
        // local_gpu_buffer_size_ = local_gpu_size;
        while (clients < max_clients) {
            rdma_cm_event* event;
            if (get_cm_event(RDMA_CM_EVENT_CONNECT_REQUEST, &event) != 0) {
                std::cerr << "[Server] get_cm_event() failed.\n";
                break;
            }

            rdma_cm_id* client_id = event->id;
            rdma_ack_cm_event(event);

            if (setup_connection(client_id) != 0) {
                std::cerr << "[Server] setup_connection() failed.\n";
                rdma_destroy_id(client_id);
                continue;
            }
            if (accept_connection(client_id) != 0) {
                rdma_destroy_id(client_id);
                continue;
            }
            // 等待连接建立完成
            if (get_cm_event(RDMA_CM_EVENT_ESTABLISHED, &event) != 0) {
                rdma_destroy_id(client_id);
                continue;
            }
            rdma_ack_cm_event(event);
            // std::cout << "[Server] Client connected" << std::endl;
            // 为每个客户端分配独立的本地 CPU 和 GPU 内存
            if (local_cpu_size > 0 && allocate_client_local_cpu_memory(client_id, local_cpu_size, hugepage) != 0) {
                std::cerr << "[Server] allocate_client_local_cpu_memory() failed.\n";
                rdma_destroy_id(client_id);
                continue;
            }
            if (local_gpu_size > 0 && allocate_client_local_gpu_memory(client_id, local_gpu_size) != 0) {
                std::cerr << "[Server] allocate_client_local_gpu_memory() failed.\n";
                rdma_destroy_id(client_id);
                continue;
            }

            // 交换元数据
            if (exchange_metadata_tcp(client_id) != 0) {
                std::cerr << "[Server] exchange_metadata() failed.\n";
                rdma_destroy_id(client_id);
                continue;
            }

            client_connections_.push_back(client_id);
            clients++;
        }
        return (clients == max_clients) ? 0 : -1;
    }

    int connect_client(int rank_id, size_t cpu_size, size_t gpu_size, bool hugepage = false) {
        // Create event channel and cm_id
        if (create_event_channel() != 0) return -1;
        if (create_id() != 0) return -1;

        // Resolve server address and route
        sockaddr_in addr = create_addr();
        if (resolve_address(&addr) != 0) return -1;
        if (resolve_route() != 0) return -1;

        // Setup QP and connect
        if (setup_connection(cm_id_) != 0) return -1;
        if (rdma_connect(cm_id_, create_conn_param()) != 0) {
            std::cerr << "[Client] rdma_connect() failed.\n";
            return -1;
        }
        // std::cout << "Client connected server" << std::endl;

        // Wait for connection establishment
        rdma_cm_event* event;
        if (get_cm_event(RDMA_CM_EVENT_ESTABLISHED, &event) != 0) {
            std::cerr << "[Client] get_cm_event() failed.\n";
            return -1;
        }
        rdma_ack_cm_event(event);
        // std::cout << "Wait for event done" << std::endl;

        // Allocate client local memory
        if (cpu_size > 0 && allocate_cpu_memory(cpu_size, hugepage) != 0) {
            std::cerr << "[Client] allocate_cpu_memory() failed.\n";
            return -1;
        }
        if (gpu_size > 0 && allocate_gpu_memory(gpu_size) != 0) {
            std::cerr << "[Client] allocate_gpu_memory() failed.\n";
            return -1;
        }

        // Set rank ID and exchange metadata
        rank_id_ = rank_id;
        return exchange_metadata_tcp(cm_id_);
    }

    /**
     * @brief Allocate CPU memory, on client
     */
    int allocate_cpu_memory(size_t size, bool hugepage = false) {
        if (!pd_) {
            std::cerr << "[Error] PD not initialized.\n";
            return -1;
        }
        char* buffer = nullptr;
        if (hugepage) {
            buffer = (char*)mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (buffer == MAP_FAILED) {
                std::cerr << "[Error] mmap() for hugepage failed.\n";
                return -1;
            }
            use_hugepage_ = true;
        } else {
            buffer = (char*)malloc(size);
            if (!buffer) {
                std::cerr << "[Error] malloc() failed.\n";
                return -1;
            }
            use_hugepage_ = false;
        }

        int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        ibv_mr* mr = ibv_reg_mr(pd_, buffer, size, access);
        if (!mr) {
            std::cerr << "[Error] ibv_reg_mr() failed.\n";
            if (use_hugepage_) munmap(buffer, size);
            else free(buffer);
            return -1;
        }
        local_cpu_mr_ = mr;
        local_cpu_buffer_ = buffer;
        local_cpu_buffer_size_ = size;
        local_cpu_allocated_ = true;
        return 0;
    }

    /**
     * @brief Allocate GPU memory, on client
     */
    int allocate_gpu_memory(size_t size) {
        if (!pd_) {
            std::cerr << "[Error] PD not initialized.\n";
            return -1;
        }
        void* gpu_ptr;
        if (cudaMalloc(&gpu_ptr, size) != cudaSuccess) {
            std::cerr << "[Error] cudaMalloc() failed.\n";
            return -1;
        }
        int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        ibv_mr* mr = ibv_reg_mr(pd_, gpu_ptr, size, access);
        if (!mr) {
            std::cerr << "[Error] ibv_reg_mr() failed.\n";
            cudaFree(gpu_ptr);
            return -1;
        }
        local_gpu_mr_ = mr;
        local_gpu_buffer_ = static_cast<char*>(gpu_ptr);
        local_gpu_buffer_size_ = size;
        local_gpu_allocated_ = true;
        return 0;
    }

    /**
     * @brief Exchange metadata between server and client
     */
    int exchange_metadata(rdma_cm_id* id) {
        // 定义元数据结构
        struct Metadata {
            uint64_t addr;      // 内存地址
            uint32_t rkey;      // 远程密钥
            size_t size;        // 内存大小
            char type[4];       // "cpu" 或 "gpu"
            bool allocated;     // 是否已分配
            int rank_id;        // 客户端的rank ID
        };
    
        if (mode_ == "server") {
            // 服务器端：只接收客户端的MR信息
            Metadata client_meta[2]; // 假设客户端发送CPU和GPU的MR信息
            ibv_mr* recv_mr = ibv_reg_mr(pd_, client_meta, sizeof(client_meta), IBV_ACCESS_LOCAL_WRITE);
            if (!recv_mr) {
                std::cerr << "[exchange_metadata] ibv_reg_mr(recv_mr) failed.\n";
                return -1;
            }
    
            // 接收客户端的元数据
            if (post_recv(id, recv_mr, sizeof(client_meta)) != 0 || poll_completion(id) != 0) {
                std::cerr << "[exchange_metadata] Receive failed.\n";
                ibv_dereg_mr(recv_mr);
                return -1;
            }
    
            // 存储客户端的MR信息
            if (client_meta[0].allocated) {
                client_cpu_info_map_[id] = {client_meta[0].addr, client_meta[0].rkey, 
                                            client_meta[0].size, std::string(client_meta[0].type), true};
            }
            if (client_meta[1].allocated) {
                client_gpu_info_map_[id] = {client_meta[1].addr, client_meta[1].rkey, 
                                            client_meta[1].size, std::string(client_meta[1].type), true};
            }
            client_rank_map_[client_meta[0].rank_id] = id;
    
            // 清理接收MR
            ibv_dereg_mr(recv_mr);
        } else {
            // 客户端：发送自己的MR信息
            Metadata local_metas[2] = {0};
            if (local_cpu_allocated_) {
                local_metas[0].addr = (uint64_t)local_cpu_buffer_;
                local_metas[0].rkey = local_cpu_mr_->rkey;
                local_metas[0].size = local_cpu_buffer_size_;
                strncpy(local_metas[0].type, "cpu", 4);
                local_metas[0].allocated = true;
                local_metas[0].rank_id = rank_id_;
            }
            if (local_gpu_allocated_) {
                local_metas[1].addr = (uint64_t)local_gpu_buffer_;
                local_metas[1].rkey = local_gpu_mr_->rkey;
                local_metas[1].size = local_gpu_buffer_size_;
                strncpy(local_metas[1].type, "gpu", 4);
                local_metas[1].allocated = true;
                local_metas[1].rank_id = rank_id_;
            }
    
            ibv_mr* send_mr = ibv_reg_mr(pd_, local_metas, sizeof(local_metas), IBV_ACCESS_LOCAL_WRITE);
            if (!send_mr) {
                std::cerr << "[exchange_metadata] ibv_reg_mr(send_mr) failed.\n";
                return -1;
            }
    
            // 发送元数据给服务器
            if (post_send(id, send_mr, sizeof(local_metas)) != 0 || poll_completion(id) != 0) {
                std::cerr << "[exchange_metadata] Send failed.\n";
                ibv_dereg_mr(send_mr);
                return -1;
            }
    
            // 清理发送MR
            ibv_dereg_mr(send_mr);
        }
        return 0;
    }


    int exchange_metadata_tcp(rdma_cm_id* id) {
        const int META_TIMEOUT_SEC = 5;   // 元数据交换超时时间
        const int MAX_RETRIES = 10;        // 最大重试次数
    
        int meta_port = std::stoi(port_) + 1000; // 假设元数据交换端口为原端口+1000
        
        if (mode_ == "server") {
            /******************** 服务器端实现 ********************/
            int sockfd = socket(AF_INET, SOCK_STREAM, 0);
            if (sockfd < 0) {
                std::cerr << "[Meta] TCP socket failed: " << strerror(errno) << std::endl;
                return -1;
            }
    
            // 设置端口复用和超时
            int opt = 1;
            setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            struct timeval tv{ META_TIMEOUT_SEC, 0 };
            setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    
            // 绑定监听
            sockaddr_in serv_addr{};
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_addr.s_addr = INADDR_ANY;
            serv_addr.sin_port = htons(meta_port);
            
            if (bind(sockfd, (sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
                std::cerr << "[Meta] Bind failed: " << strerror(errno) 
                         << " port=" << meta_port << std::endl;
                close(sockfd);
                return -1;
            }
    
            if (listen(sockfd, 1) < 0) {
                std::cerr << "[Meta] Listen failed: " << strerror(errno) << std::endl;
                close(sockfd);
                return -1;
            }
            const int ACCEPT_RETRIES = 10; // 最大重试次数
            int retry_count = 0;
            int connfd;
            while (retry_count < ACCEPT_RETRIES) {
                // std::cout << "[Meta] Waiting connection (attempt " << (retry_count+1) << ")...\n";
                connfd = accept(sockfd, nullptr, nullptr);
                if (connfd >= 0) {
                    // 成功接受连接
                    break;
                } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // 超时，继续重试
                    retry_count++;
                    continue;
                } else {
                    // 其他错误直接退出
                    std::cerr << "[Meta] Accept failed: " << strerror(errno) << std::endl;
                    close(sockfd);
                    return -1;
                }
            }
            
            if (retry_count >= ACCEPT_RETRIES) {
                std::cerr << "[Meta] Accept timeout after " << ACCEPT_RETRIES << " attempts\n";
                close(sockfd);
                return -1;
            }    
            // // 接受连接（带超时）
            // std::cout << "[Meta] Waiting connection on port " << meta_port << "...\n";
            // int connfd = accept(sockfd, nullptr, nullptr);
            // if (connfd < 0) {
            //     std::cerr << "[Meta] Accept failed: " << strerror(errno) << std::endl;
            //     close(sockfd);
            //     return -1;
            // }
    
            // 接收元数据（处理分包）
            Metadata client_meta[2]{};
            char* buf = reinterpret_cast<char*>(client_meta);
            size_t total = 0, remaining = sizeof(client_meta);
    
            while (remaining > 0) {
                ssize_t recvd = recv(connfd, buf + total, remaining, 0);
                if (recvd <= 0) {
                    std::cerr << "[Meta] Recv failed: " 
                             << (recvd == 0 ? "Connection closed" : strerror(errno)) 
                             << std::endl;
                    close(connfd);
                    close(sockfd);
                    return -1;
                }
                total += recvd;
                remaining -= recvd;
            }
    
            // 存储元数据
            if (client_meta[0].allocated) {
                client_cpu_info_map_[id] = {
                    client_meta[0].addr, client_meta[0].rkey,
                    client_meta[0].size, std::string(client_meta[0].type, 3),
                    true
                };
                std::cout << "[Meta] Received CPU meta: addr=0x" << std::hex << client_meta[0].addr 
                         << " rkey=" << client_meta[0].rkey << std::dec << std::endl;
            }
    
            if (client_meta[1].allocated) {
                client_gpu_info_map_[id] = {
                    client_meta[1].addr, client_meta[1].rkey,
                    client_meta[1].size, std::string(client_meta[1].type, 3),
                    true
                };
                std::cout << "[Meta] Received GPU meta: addr=0x" << std::hex << client_meta[1].addr 
                         << " rkey=" << client_meta[1].rkey << std::dec << std::endl;
            }
    
            client_rank_map_[client_meta[0].rank_id] = id;
            close(connfd);
            close(sockfd);
            return 0;
    
        } else {
            /******************** 客户端实现 ********************/
            int retries = 0;
            while (retries < MAX_RETRIES) {
                int sockfd = socket(AF_INET, SOCK_STREAM, 0);
                if (sockfd < 0) {
                    std::cerr << "[Meta] Client socket failed: " << strerror(errno) << std::endl;
                    return -1;
                }
    
                // 设置连接超时
                timeval tv{ META_TIMEOUT_SEC, 0 };
                setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    
                sockaddr_in serv_addr{};
                serv_addr.sin_family = AF_INET;
                serv_addr.sin_port = htons(meta_port);
                // std::cout<<"client listen to "<<ip_<<std::endl;
                if (inet_pton(AF_INET, ip_.c_str(), &serv_addr.sin_addr) <= 0) {
                    std::cerr << "[Meta] Invalid server IP: " << ip_ << std::endl;
                    close(sockfd);
                    return -1;
                }
    
                // 尝试连接
                if (connect(sockfd, (sockaddr*)&serv_addr, sizeof(serv_addr)) == 0) {
                    // 准备元数据
                    Metadata local_metas[2]{};
                    if (local_cpu_allocated_) {
                        local_metas[0] = {
                            reinterpret_cast<uint64_t>(local_cpu_buffer_),
                            local_cpu_mr_->rkey,
                            local_cpu_buffer_size_,
                            {'c','p','u',0},  // 显式初始化char数组
                            true,
                            rank_id_
                        };
                    }
    
                    if (local_gpu_allocated_) {
                        local_metas[1] = {
                            reinterpret_cast<uint64_t>(local_gpu_buffer_),
                            local_gpu_mr_->rkey,
                            local_gpu_buffer_size_,
                            {'g','p','u',0},
                            true,
                            rank_id_
                        };
                    }
    
                    // 发送数据（处理分块）
                    const char* data = reinterpret_cast<const char*>(local_metas);
                    size_t total = 0, remaining = sizeof(local_metas);
    
                    while (remaining > 0) {
                        ssize_t sent = send(sockfd, data + total, remaining, 0);
                        if (sent <= 0) {
                            std::cerr << "[Meta] Send failed: " 
                                     << (sent == 0 ? "Connection closed" : strerror(errno)) 
                                     << std::endl;
                            close(sockfd);
                            return -1;
                        }
                        total += sent;
                        remaining -= sent;
                    }
    
                    // std::cout << "[Meta] Sent metadata successfully\n";
                    close(sockfd);
                    return 0;
                }
    
                // 连接失败处理
                std::cerr << "[Meta] Connect to " << ip_ << ":" << meta_port 
                         << " failed (attempt " << (retries+1) << "/" << MAX_RETRIES << ")\n";
                close(sockfd);
                ++retries;
                sleep(1);  // 等待重试
            }
    
            std::cerr << "[Meta] Max retries exceeded\n";
            return -1;
        }
    }

    int post_rdma_write(int client_rank, size_t size, const std::string& src_type, 
                    const std::string& dst_type, uint64_t local_offset = 0, uint64_t remote_offset = 0) {
        if (mode_ != "server") {
            std::cerr << "[Error] Only server can initiate RDMA write.\n";
            return -1;
        }
        auto it = client_rank_map_.find(client_rank);
        if (it == client_rank_map_.end()) {
            std::cerr << "[post_rdma_write] Client rank " << client_rank << " not found.\n";
            return -1;
        }
        rdma_cm_id* id = it->second;

        // 选择该客户端的源缓冲区和 MR
        char* src_buffer = (src_type == "cpu") ? client_local_cpu_buffer_map_[id] : 
                        (src_type == "gpu") ? client_local_gpu_buffer_map_[id] : nullptr;
        ibv_mr* src_mr = (src_type == "cpu") ? client_local_cpu_mr_map_[id] : 
                        (src_type == "gpu") ? client_local_gpu_mr_map_[id] : nullptr;
        size_t src_size = (src_type == "cpu") ? client_local_cpu_buffer_size_map_[id] : 
                        (src_type == "gpu") ? client_local_gpu_buffer_size_map_[id] : 0;
        if (!src_buffer || (local_offset + size) > src_size) {
            std::cerr << "[post_rdma_write] Invalid source memory for client.\n";
            return -1;
        }

        // 应用本地偏移量
        char* adjusted_src_buffer = src_buffer + local_offset;

        // 选择目标信息
        RemoteMemoryInfo* dst_info = (dst_type == "cpu") ? &client_cpu_info_map_[id] : 
                                    (dst_type == "gpu") ? &client_gpu_info_map_[id] : nullptr;
        if (!dst_info || !dst_info->allocated || (remote_offset + size) > dst_info->size) {
            std::cerr << "[post_rdma_write] Invalid destination memory.\n";
            return -1;
        }

        ibv_sge sge = {(uintptr_t)adjusted_src_buffer, static_cast<uint32_t>(size), src_mr->lkey};
        ibv_send_wr wr = {0};
        wr.wr_id = (uintptr_t)id;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.wr.rdma.remote_addr = dst_info->addr + remote_offset;
        wr.wr.rdma.rkey = dst_info->rkey;

        ibv_send_wr* bad_wr = nullptr;
        if (ibv_post_send(id->qp, &wr, &bad_wr) != 0) {
            std::cerr << "[post_rdma_write] ibv_post_send failed.\n";
            return -1;
        }
        return poll_completion(id);
    }

    int post_rdma_read(int client_rank, size_t size, const std::string& src_type, 
        const std::string& dst_type, uint64_t local_offset = 0, uint64_t remote_offset = 0) {
        if (mode_ != "server") {
            std::cerr << "[Error] Only server can initiate RDMA read.\n";
            return -1;
        }
        auto it = client_rank_map_.find(client_rank);
        if (it == client_rank_map_.end()) {
            std::cerr << "[post_rdma_read] Client rank " << client_rank << " not found.\n";
            return -1;
        }
        rdma_cm_id* id = it->second;

        // 选择该客户端的目标缓冲区和 MR
        char* dst_buffer = (dst_type == "cpu") ? client_local_cpu_buffer_map_[id] : 
                    (dst_type == "gpu") ? client_local_gpu_buffer_map_[id] : nullptr;
        ibv_mr* dst_mr = (dst_type == "cpu") ? client_local_cpu_mr_map_[id] : 
                (dst_type == "gpu") ? client_local_gpu_mr_map_[id] : nullptr;
        size_t dst_size = (dst_type == "cpu") ? client_local_cpu_buffer_size_map_[id] : 
                (dst_type == "gpu") ? client_local_gpu_buffer_size_map_[id] : 0;
        if (!dst_buffer || (local_offset + size) > dst_size) {
            std::cerr << "[post_rdma_read] Invalid destination memory for client.\n";
            return -1;
        }

        // 应用本地偏移量
        char* adjusted_dst_buffer = dst_buffer + local_offset;

        // 选择源信息
        RemoteMemoryInfo* src_info = (src_type == "cpu") ? &client_cpu_info_map_[id] : 
                            (src_type == "gpu") ? &client_gpu_info_map_[id] : nullptr;
        if (!src_info || !src_info->allocated || (remote_offset + size) > src_info->size) {
            std::cerr << "[post_rdma_read] Invalid source memory.\n";
            return -1;
        }

        ibv_sge sge = {(uintptr_t)adjusted_dst_buffer, static_cast<uint32_t>(size), dst_mr->lkey};
        ibv_send_wr wr = {0};
        wr.wr_id = (uintptr_t)id;
        wr.opcode = IBV_WR_RDMA_READ;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.wr.rdma.remote_addr = src_info->addr + remote_offset;
        wr.wr.rdma.rkey = src_info->rkey;

        ibv_send_wr* bad_wr = nullptr;
        if (ibv_post_send(id->qp, &wr, &bad_wr) != 0) {
            std::cerr << "[post_rdma_read] ibv_post_send failed.\n";
            return -1;
        }
        int ret = poll_completion(id);
        return ret;
    }

    /**
     * @brief Get local CPU buffer as tensor
     */
    torch::Tensor get_client_cpu_tensor() {
        if (!local_cpu_allocated_) {
            throw std::runtime_error("Local CPU memory not allocated.");
        }
        return torch::from_blob(local_cpu_buffer_, {static_cast<long>(local_cpu_buffer_size_)}, 
                                torch::kUInt8);
    }

    /**
     * @brief Get local GPU buffer as tensor
     */
    torch::Tensor get_client_gpu_tensor() {
        if (!local_gpu_allocated_) {
            throw std::runtime_error("Local GPU memory not allocated.");
        }
        return torch::from_blob(local_gpu_buffer_, {static_cast<long>(local_gpu_buffer_size_)}, 
                                torch::dtype(torch::kUInt8).device(torch::kCUDA));
    }

    torch::Tensor get_server_cpu_tensor(int rank) {
        auto it = client_rank_map_.find(rank);
        if (it == client_rank_map_.end()) {
            throw std::runtime_error("Client rank not found.");
        }
        rdma_cm_id* id = it->second;
        if (client_local_cpu_buffer_map_.find(id) == client_local_cpu_buffer_map_.end()) {
            throw std::runtime_error("Client read buffer not allocated.");
        }
        char* buffer = client_local_cpu_buffer_map_[id];
        size_t size = client_local_cpu_buffer_size_map_[id];
        return torch::from_blob(buffer, {static_cast<long>(size)}, torch::kChar);
    }    

    torch::Tensor get_server_gpu_tensor(int rank) {
        auto it = client_rank_map_.find(rank);
        if (it == client_rank_map_.end()) {
            throw std::runtime_error("Client rank not found.");
        }
        rdma_cm_id* id = it->second;
        if (client_local_gpu_buffer_map_.find(id) == client_local_gpu_buffer_map_.end()) {
            throw std::runtime_error("Client read buffer not allocated.");
        }
        char* buffer = client_local_gpu_buffer_map_[id];
        size_t size = client_local_gpu_buffer_size_map_[id];
        return torch::from_blob(buffer, {static_cast<long>(size)}, 
                                    torch::dtype(torch::kUInt8).device(torch::kCUDA));
    }   
    
    // size_t get_cpu_buffer_size() {
    //     return local_cpu_buffer_size_;
    // }

    // size_t get_gpu_buffer_size() {
    //     return local_gpu_buffer_size_;
    // }

private:
    int allocate_client_local_cpu_memory(rdma_cm_id* client_id, size_t size, bool hugepage) {
        char* buffer = nullptr;
        if (hugepage) {
            buffer = (char*)mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (buffer == MAP_FAILED) {
                std::cerr << "[Error] mmap() for client local CPU memory failed.\n";
                return -1;
            }
        } else {
            buffer = (char*)malloc(size);
            if (!buffer) {
                std::cerr << "[Error] malloc() for client local CPU memory failed.\n";
                return -1;
            }
        }

        int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        ibv_mr* mr = ibv_reg_mr(pd_, buffer, size, access);
        if (!mr) {
            std::cerr << "[Error] ibv_reg_mr() for client local CPU memory failed.\n";
            if (hugepage) munmap(buffer, size);
            else free(buffer);
            return -1;
        }

        client_local_cpu_mr_map_[client_id] = mr;
        client_local_cpu_buffer_map_[client_id] = buffer;
        client_local_cpu_buffer_size_map_[client_id] = size;
        return 0;
    }

    int allocate_client_local_gpu_memory(rdma_cm_id* client_id, size_t size) {
        void* gpu_ptr;
        if (cudaMalloc(&gpu_ptr, size) != cudaSuccess) {
            std::cerr << "[Error] cudaMalloc() for client local GPU memory failed.\n";
            return -1;
        }
        int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        ibv_mr* mr = ibv_reg_mr(pd_, gpu_ptr, size, access);
        if (!mr) {
            std::cerr << "[Error] ibv_reg_mr() for client local GPU memory failed.\n";
            cudaFree(gpu_ptr);
            return -1;
        }
        client_local_gpu_mr_map_[client_id] = mr;
        client_local_gpu_buffer_map_[client_id] = static_cast<char*>(gpu_ptr);
        client_local_gpu_buffer_size_map_[client_id] = size;
        return 0;
    }

    void cleanup() {
        // 释放客户端的本地内存
        for (auto& [id, mr] : client_local_cpu_mr_map_) {
            ibv_dereg_mr(mr);
            if (use_hugepage_) munmap(client_local_cpu_buffer_map_[id], client_local_cpu_buffer_size_map_[id]);
            else free(client_local_cpu_buffer_map_[id]);
        }
        for (auto& [id, mr] : client_local_gpu_mr_map_) {
            ibv_dereg_mr(mr);
            cudaFree(client_local_gpu_buffer_map_[id]);
        }
        client_local_cpu_mr_map_.clear();
        client_local_gpu_mr_map_.clear();
        client_local_cpu_buffer_map_.clear();
        client_local_gpu_buffer_map_.clear();
        client_local_cpu_buffer_size_map_.clear();
        client_local_gpu_buffer_size_map_.clear();
    
        // 原有的清理逻辑保持不变
        if (local_cpu_mr_) {
            ibv_dereg_mr(local_cpu_mr_);
            if (use_hugepage_) munmap(local_cpu_buffer_, local_cpu_buffer_size_);
            else free(local_cpu_buffer_);
            local_cpu_mr_ = nullptr;
            local_cpu_buffer_ = nullptr;
        }
        if (local_gpu_mr_) {
            ibv_dereg_mr(local_gpu_mr_);
            cudaFree(local_gpu_buffer_);
            local_gpu_mr_ = nullptr;
            local_gpu_buffer_ = nullptr;
        }
        for (auto id : client_connections_) {
            if (client_cqs_[id]) ibv_destroy_cq(client_cqs_[id]);
            rdma_destroy_qp(id);
            rdma_destroy_id(id);
        }
        client_connections_.clear();
        client_cqs_.clear();
        if (pd_) {
            ibv_dealloc_pd(pd_);
            pd_ = nullptr;
        }
        if (cm_id_) {
            rdma_destroy_id(cm_id_);
            cm_id_ = nullptr;
        }
        if (ec_) {
            rdma_destroy_event_channel(ec_);
            ec_ = nullptr;
        }
    }
    /**
     * @brief Post send operation
     */
    int post_send(rdma_cm_id* id, ibv_mr* mr, size_t size) {
        ibv_sge sge = {(uintptr_t)mr->addr, static_cast<uint32_t>(size), mr->lkey};
        ibv_send_wr wr = {0};
        wr.wr_id = (uintptr_t)id;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        ibv_send_wr* bad_wr = nullptr;
        return ibv_post_send(id->qp, &wr, &bad_wr);
    }

    /**
     * @brief Post receive operation
     */
    int post_recv(rdma_cm_id* id, ibv_mr* mr, size_t size) {
        ibv_sge sge = {(uintptr_t)mr->addr, static_cast<uint32_t>(size), mr->lkey};
        ibv_recv_wr wr = {0};
        wr.wr_id = (uintptr_t)id;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        ibv_recv_wr* bad_wr = nullptr;
        return ibv_post_recv(id->qp, &wr, &bad_wr);
    }

    /**
     * @brief Poll completion queue
     */
    int poll_completion(rdma_cm_id* id) {
        auto it = client_cqs_.find(id);
        if (it == client_cqs_.end()) {
            std::cerr << "[poll_completion] CQ not found.\n";
            return -1;
        }
        ibv_cq* cq = it->second;
        ibv_wc wc;
        int ret;
        do {
            ret = ibv_poll_cq(cq, 1, &wc);
        } while (ret == 0);
        if (ret < 0 || wc.status != IBV_WC_SUCCESS) {
            std::cerr << "[poll_completion] Failed: ret=" << ret << ", status=" << wc.status << "\n";
            return -1;
        }
        return 0;
    }

    int create_event_channel() {
        ec_ = rdma_create_event_channel();
        if (!ec_) {
            std::cerr << "rdma_create_event_channel failed: " << strerror(errno) << "\n";
            return -1;
        }
        return 0;
    }

    int create_id() {
        if (rdma_create_id(ec_, &cm_id_, nullptr, RDMA_PS_TCP)) {
            std::cerr << "rdma_create_id failed: " << strerror(errno) << "\n";
            return -1;
        }
        return 0;
    }

    sockaddr_in create_addr(in_addr_t addr = INADDR_NONE) {
        sockaddr_in sa = {0};
        sa.sin_family = AF_INET;
        sa.sin_port = htons(std::stoi(port_));
        if (addr == INADDR_NONE) {
            if (inet_pton(AF_INET, ip_.c_str(), &sa.sin_addr) != 1) {
                std::cerr << "inet_pton failed for IP: " << ip_ << "\n";
            }
        } else {
            sa.sin_addr.s_addr = addr;
        }
        return sa;
    }

    int bind_and_listen(sockaddr_in* addr) {
        if (rdma_bind_addr(cm_id_, (sockaddr*)addr)) {
            std::cerr << "rdma_bind_addr failed: " << strerror(errno) << "\n";
            return -1;
        }
        if (rdma_listen(cm_id_, 10)) {
            std::cerr << "rdma_listen failed: " << strerror(errno) << "\n";
            return -1;
        }
        // std::cout << "Listening on port " << port_ << "\n";
        return 0;
    }

    int resolve_address(sockaddr_in* addr) {
        if (rdma_resolve_addr(cm_id_, nullptr, (sockaddr*)addr, 2000)) {
            std::cerr << "rdma_resolve_addr failed: " << strerror(errno) << "\n";
            return -1;
        }
        return wait_for_event(RDMA_CM_EVENT_ADDR_RESOLVED);
    }

    int resolve_route() {
        if (rdma_resolve_route(cm_id_, 2000)) {
            std::cerr << "rdma_resolve_route failed: " << strerror(errno) << "\n";
            return -1;
        }
        return wait_for_event(RDMA_CM_EVENT_ROUTE_RESOLVED);
    }

    int setup_connection(rdma_cm_id* id) {
        if (!pd_) {
            pd_ = ibv_alloc_pd(id->verbs);
            if (!pd_) {
                std::cerr << "ibv_alloc_pd failed: " << strerror(errno) << "\n";
                return -1;
            }
        }
        ibv_cq* cq = ibv_create_cq(id->verbs, 10, nullptr, nullptr, 0);
        if (!cq) {
            std::cerr << "ibv_create_cq failed: " << strerror(errno) << "\n";
            return -1;
        }
        ibv_qp_init_attr qp_attr = {0};
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.send_cq = cq;
        qp_attr.recv_cq = cq;
        qp_attr.qp_type = IBV_QPT_RC;
        if (rdma_create_qp(id, pd_, &qp_attr)) {
            std::cerr << "rdma_create_qp failed: " << strerror(errno) << "\n";
            ibv_destroy_cq(cq);
            return -1;
        }
        client_cqs_[id] = cq;
        return 0;
    }

    rdma_conn_param* create_conn_param() {
        static rdma_conn_param param = {0};
        param.initiator_depth = 1;
        param.responder_resources = 1;
        param.rnr_retry_count = 7;
        return &param;
    }

    int get_cm_event(rdma_cm_event_type expected, rdma_cm_event** event) {
        if (rdma_get_cm_event(ec_, event)) {
            std::cerr << "rdma_get_cm_event failed: " << strerror(errno) << "\n";
            return -1;
        }
        if ((*event)->event != expected) {
            std::cerr << "Unexpected event: " << rdma_event_str((*event)->event) << "\n";
            rdma_ack_cm_event(*event);
            return -1;
        }
        return 0;
    }

    int accept_connection(rdma_cm_id* client_id) {
        rdma_conn_param param;
        memset(&param, 0, sizeof(param));
        param.initiator_depth = 1;
        param.responder_resources = 1;
        param.rnr_retry_count = 7;
        if (rdma_accept(client_id, &param)) {
            perror("rdma_accept");
            return -1;
        }
        // std::cout << "Server accepted connection" << std::endl;
        return 0;
    }

    int wait_for_event(rdma_cm_event_type expected) {
        rdma_cm_event* event;
        if (get_cm_event(expected, &event)) return -1;
        rdma_ack_cm_event(event);
        return 0;
    }

    bool check_rdma_device() {
        ibv_device** dev_list = ibv_get_device_list(nullptr);
        if (!dev_list) {
            std::cerr << "[Error] No RDMA devices found.\n";
            return false;
        }
        ibv_free_device_list(dev_list);
        return true;
    }

    bool check_network_config() {
        if (ip_.empty() || port_.empty()) {
            std::cerr << "[Error] IP or port is empty.\n";
            return false;
        }
        return true;
    }

    // Members
    std::string ip_;
    std::string port_;
    std::string mode_;

    rdma_event_channel* ec_;
    rdma_cm_id* cm_id_;
    ibv_pd* pd_;

    std::map<rdma_cm_id*, ibv_mr*> client_local_cpu_mr_map_;
    std::map<rdma_cm_id*, ibv_mr*> client_local_gpu_mr_map_;
    std::map<rdma_cm_id*, char*> client_local_cpu_buffer_map_;
    std::map<rdma_cm_id*, char*> client_local_gpu_buffer_map_;
    std::map<rdma_cm_id*, size_t> client_local_cpu_buffer_size_map_;
    std::map<rdma_cm_id*, size_t> client_local_gpu_buffer_size_map_;

    ibv_mr* local_cpu_mr_;
    ibv_mr* local_gpu_mr_;
    char* local_cpu_buffer_;
    char* local_gpu_buffer_;
    size_t local_cpu_buffer_size_;
    size_t local_gpu_buffer_size_;
    bool local_cpu_allocated_;
    bool local_gpu_allocated_;
    bool use_hugepage_;

    int rank_id_;

    std::vector<rdma_cm_id*> client_connections_;
    std::map<rdma_cm_id*, ibv_cq*> client_cqs_;
    std::map<int, rdma_cm_id*> client_rank_map_;

    // Remote memory info
    std::map<rdma_cm_id*, RemoteMemoryInfo> client_cpu_info_map_;
    std::map<rdma_cm_id*, RemoteMemoryInfo> client_gpu_info_map_;
    RemoteMemoryInfo server_read_info_;
    RemoteMemoryInfo server_write_info_;
};

#endif // RDMA_ONESIDED_ENDPOINT_H