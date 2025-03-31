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

/**
 * @brief 存储远端内存的信息
 */
struct RemoteMemoryInfo {
    uint64_t addr;    // 远程内存地址
    uint32_t rkey;    // 远程内存访问密钥
    size_t size;      // 内存区域大小
};

/**
 * @brief RDMAOneSidedEndpoint
 * 用于封装 RDMA 连接、单边读写操作以及元数据交换流程
 */
class RDMAOneSidedEndpoint {
public:
    RDMAOneSidedEndpoint(const std::string& ip, 
                         const std::string& port, 
                         const std::string& mode)
        : ip_(ip), port_(port), mode_(mode), 
          ec_(nullptr), cm_id_(nullptr), pd_(nullptr), 
          local_mr_(nullptr), local_buffer_(nullptr), 
          buffer_size_(0), use_hugepage_(false) {}

    ~RDMAOneSidedEndpoint() {
        cleanup();
    }

    /**
     * @brief 服务器端初始化并等待客户端连接
     * @param max_clients 最大客户端连接数
     * @param mem_size    本地需要注册的内存大小
     * @param hugepage    是否使用大页
     */
    int run_server(int max_clients, size_t mem_size, bool hugepage = false) {
        // 1) 创建事件通道和 cm_id
        if (create_event_channel() != 0) return -1;
        if (create_id() != 0) return -1;

        // 2) 绑定地址并监听
        sockaddr_in addr = create_addr(INADDR_ANY);
        if (bind_and_listen(&addr) != 0) return -1;

        // 3) 注册本地内存 (服务端可提前分配共享区域)
        if (register_memory(mem_size, hugepage) != 0) {
            std::cerr << "[Server] register_memory() failed.\n";
            return -1;
        }

        // 4) 等待客户端连接
        int clients = 0;
        while (clients < max_clients) {
            rdma_cm_event* event;
            // 等待 CONNECT_REQUEST
            if (get_cm_event(RDMA_CM_EVENT_CONNECT_REQUEST, &event) != 0) {
                std::cerr << "[Server] get_cm_event() failed.\n";
                break;
            }

            rdma_cm_id* client_id = event->id;
            rdma_ack_cm_event(event);

            // 建立 RDMA QP 连接
            if (setup_connection(client_id) != 0) {
                std::cerr << "[Server] setup_connection() failed.\n";
                rdma_destroy_id(client_id);
                continue;
            }

            // 和客户端交换元数据
            if (exchange_metadata(client_id) != 0) {
                std::cerr << "[Server] exchange_metadata() failed.\n";
                rdma_destroy_id(client_id);
                continue;
            }

            client_connections_.push_back(client_id);
            clients++;
        }
        return (clients == max_clients) ? 0 : -1;
    }

    /**
     * @brief 客户端连接到服务器
     * @param mem_size  本地需要注册的内存大小
     * @param hugepage  是否使用大页
     */
    int connect_client(size_t mem_size, bool hugepage = false) {
        // 1) 创建事件通道和 cm_id
        if (create_event_channel() != 0) return -1;
        if (create_id() != 0) return -1;

        // 2) 解析服务器地址并路由
        sockaddr_in addr = create_addr();
        if (resolve_address(&addr) != 0) return -1;
        if (resolve_route() != 0) return -1;

        // 3) 创建 QP, 发起连接
        if (setup_connection(cm_id_) != 0) return -1;
        if (rdma_connect(cm_id_, create_conn_param()) != 0) {
            std::cerr << "[Client] rdma_connect() failed.\n";
            return -1;
        }

        // 4) 等待 ESTABLISHED 事件
        rdma_cm_event* event;
        if (get_cm_event(RDMA_CM_EVENT_ESTABLISHED, &event) != 0) {
            std::cerr << "[Client] get_cm_event() failed.\n";
            return -1;
        }
        rdma_ack_cm_event(event);

        // 5) 在客户端注册本地内存
        if (register_memory(mem_size, hugepage) != 0) {
            std::cerr << "[Client] register_memory() failed.\n";
            return -1;
        }

        // 6) 交换元数据 (与服务器进行 addr、rkey 交互)
        return exchange_metadata(cm_id_);
    }

    /**
     * @brief 注册本地缓冲区，创建 MR
     * @param size     缓冲区大小
     * @param hugepage 是否使用大页分配
     */
    int register_memory(size_t size, bool hugepage = false) {
        if (!pd_) {
            std::cerr << "[Error] PD not initialized, cannot register memory!\n";
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

        // 注册 MR，设置远程可读写
        int access = IBV_ACCESS_LOCAL_WRITE
                   | IBV_ACCESS_REMOTE_READ
                   | IBV_ACCESS_REMOTE_WRITE;

        ibv_mr* mr = ibv_reg_mr(pd_, buffer, size, access);
        if (!mr) {
            std::cerr << "[Error] ibv_reg_mr() failed.\n";
            if (use_hugepage_) munmap(buffer, size);
            else free(buffer);
            return -1;
        }

        local_mr_ = mr;
        local_buffer_ = buffer;
        buffer_size_ = size;
        return 0;
    }

    /**
     * @brief RDMA 单边写
     * @param id           对端连接 ID
     * @param size         写入大小
     * @param remote_offset 远端偏移
     */
    int post_rdma_write(rdma_cm_id* id, size_t size, uint64_t remote_offset = 0) {
        RemoteMemoryInfo& remote = remote_info_[id];
        if ((remote_offset + size) > remote.size) {
            std::cerr << "[post_rdma_write] Out of remote range!\n";
            return -1;
        }

        ibv_sge sge;
        sge.addr   = (uintptr_t)local_buffer_;
        sge.length = size;
        sge.lkey   = local_mr_->lkey;

        ibv_send_wr wr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id      = (uintptr_t)id;
        wr.opcode     = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.sg_list    = &sge;
        wr.num_sge    = 1;

        wr.wr.rdma.remote_addr = remote.addr + remote_offset;
        wr.wr.rdma.rkey        = remote.rkey;

        ibv_send_wr* bad_wr = nullptr;
        return ibv_post_send(id->qp, &wr, &bad_wr);
    }

    /**
     * @brief RDMA 单边读
     * @param id           对端连接 ID
     * @param size         读取大小
     * @param remote_offset 远端偏移
     */
    int post_rdma_read(rdma_cm_id* id, size_t size, uint64_t remote_offset = 0) {
        RemoteMemoryInfo& remote = remote_info_[id];
        if ((remote_offset + size) > remote.size) {
            std::cerr << "[post_rdma_read] Out of remote range!\n";
            return -1;
        }

        ibv_sge sge;
        sge.addr   = (uintptr_t)local_buffer_;
        sge.length = size;
        sge.lkey   = local_mr_->lkey;

        ibv_send_wr wr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id      = (uintptr_t)id;
        wr.opcode     = IBV_WR_RDMA_READ;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.sg_list    = &sge;
        wr.num_sge    = 1;

        wr.wr.rdma.remote_addr = remote.addr + remote_offset;
        wr.wr.rdma.rkey        = remote.rkey;

        ibv_send_wr* bad_wr = nullptr;
        return ibv_post_send(id->qp, &wr, &bad_wr);
    }

    /**
     * @brief 轮询 CQ 完成，用于等待读写操作完成
     */
    int poll_completion(rdma_cm_id* id) {
        if (client_cqs_.find(id) == client_cqs_.end()) {
            std::cerr << "[poll_completion] CQ not found!\n";
            return -1;
        }

        ibv_cq* cq = client_cqs_[id];
        ibv_wc wc;
        int ret;
        do {
            ret = ibv_poll_cq(cq, 1, &wc);
        } while (ret == 0); // 一直阻塞式轮询，直到有完成事件

        if (ret < 0) {
            std::cerr << "[poll_completion] ibv_poll_cq() < 0.\n";
            return -1;
        }
        if (wc.status != IBV_WC_SUCCESS) {
            std::cerr << "[poll_completion] WC status=" << wc.status << "\n";
            return -1;
        }
        return 0;
    }

    /**
     * @brief 获取本地缓冲区指针
     */
    char* get_local_buffer() { return local_buffer_; }

    /**
     * @brief 获取本地缓冲区对应的 Torch 张量
     */
    torch::Tensor get_local_tensor() {
        // 建议 clone 一份，防止原buffer被释放导致非法访问
        return torch::from_blob(local_buffer_, {static_cast<long>(buffer_size_)}, 
                                torch::kUInt8).clone();
    }

private:
    /**
     * @brief 释放资源
     */
    void cleanup() {
        // 注销 MR
        if (local_mr_) {
            ibv_dereg_mr(local_mr_);
            local_mr_ = nullptr;
        }

        // 释放本地缓冲区
        if (local_buffer_) {
            if (use_hugepage_) {
                munmap(local_buffer_, buffer_size_);
            } else {
                free(local_buffer_);
            }
            local_buffer_ = nullptr;
        }

        // 销毁客户端连接
        for (auto id : client_connections_) {
            rdma_destroy_id(id);
        }
        client_connections_.clear();

        // 销毁自身的 cm_id 和 PD
        if (cm_id_) {
            rdma_destroy_id(cm_id_);
            cm_id_ = nullptr;
        }
        if (pd_) {
            ibv_dealloc_pd(pd_);
            pd_ = nullptr;
        }

        // 销毁事件通道
        if (ec_) {
            rdma_destroy_event_channel(ec_);
            ec_ = nullptr;
        }
    }

    /**
     * @brief 交换本地/远程的 addr 和 rkey 信息
     */
    int exchange_metadata(rdma_cm_id* id) {
        // 发送本地元数据
        struct LocalMetadata {
            uint64_t addr;
            uint32_t rkey;
            size_t size;
        } local_meta;

        local_meta.addr = (uint64_t)local_buffer_;
        local_meta.rkey = local_mr_->rkey;
        local_meta.size = buffer_size_;

        // 用临时 mr 承载元数据结构并发送
        ibv_mr* send_mr = ibv_reg_mr(pd_, &local_meta, sizeof(local_meta), IBV_ACCESS_LOCAL_WRITE);
        if (!send_mr) {
            std::cerr << "[exchange_metadata] ibv_reg_mr(send_mr) failed.\n";
            return -1;
        }
        if (post_send(id, send_mr, sizeof(local_meta))) {
            std::cerr << "[exchange_metadata] post_send() failed.\n";
            ibv_dereg_mr(send_mr);
            return -1;
        }
        if (poll_completion(id)) {
            ibv_dereg_mr(send_mr);
            return -1;
        }

        // 接收远程元数据
        LocalMetadata remote_meta;
        ibv_mr* recv_mr = ibv_reg_mr(pd_, &remote_meta, sizeof(remote_meta), IBV_ACCESS_LOCAL_WRITE);
        if (!recv_mr) {
            std::cerr << "[exchange_metadata] ibv_reg_mr(recv_mr) failed.\n";
            ibv_dereg_mr(send_mr);
            return -1;
        }

        if (post_recv(id, recv_mr, sizeof(remote_meta))) {
            std::cerr << "[exchange_metadata] post_recv() failed.\n";
            ibv_dereg_mr(recv_mr);
            ibv_dereg_mr(send_mr);
            return -1;
        }
        if (poll_completion(id)) {
            ibv_dereg_mr(recv_mr);
            ibv_dereg_mr(send_mr);
            return -1;
        }

        // 记录对端的 addr / rkey / size
        remote_info_[id] = { 
            remote_meta.addr, 
            remote_meta.rkey, 
            remote_meta.size 
        };

        // 释放临时 MR
        ibv_dereg_mr(recv_mr);
        ibv_dereg_mr(send_mr);
        return 0;
    }

    private:
    int create_event_channel() {
        ec_ = rdma_create_event_channel();
        if (!ec_) {
            std::cerr << "rdma_create_event_channel failed: " << strerror(errno) << std::endl;
            return -1;
        }
        return 0;
    }

    int create_id() {
        if (rdma_create_id(ec_, &cm_id_, nullptr, RDMA_PS_TCP)) {
            std::cerr << "rdma_create_id failed: " << strerror(errno) << std::endl;
            return -1;
        }
        return 0;
    }

    sockaddr_in create_addr(in_addr_t addr = INADDR_NONE) {
        sockaddr_in sa;
        memset(&sa, 0, sizeof(sa));
        sa.sin_family = AF_INET;
        sa.sin_port = htons(std::stoi(port_));
        
        if (addr == INADDR_NONE) {
            if (inet_pton(AF_INET, ip_.c_str(), &sa.sin_addr) != 1) {
                std::cerr << "inet_pton failed for IP: " << ip_ << std::endl;
            }
        } else {
            sa.sin_addr.s_addr = addr;
        }
        return sa;
    }

    int bind_and_listen(sockaddr_in* addr) {
        if (rdma_bind_addr(cm_id_, (sockaddr*)addr)) {
            std::cerr << "rdma_bind_addr failed: " << strerror(errno) << std::endl;
            return -1;
        }
        if (rdma_listen(cm_id_, 10)) {
            std::cerr << "rdma_listen failed: " << strerror(errno) << std::endl;
            return -1;
        }
        std::cout << "Listening on port " << port_ << std::endl;
        return 0;
    }

    int resolve_address(sockaddr_in* addr) {
        if (rdma_resolve_addr(cm_id_, nullptr, (sockaddr*)addr, 2000)) {
            std::cerr << "rdma_resolve_addr failed: " << strerror(errno) << std::endl;
            return -1;
        }
        return wait_for_event(RDMA_CM_EVENT_ADDR_RESOLVED);
    }

    int resolve_route() {
        if (rdma_resolve_route(cm_id_, 2000)) {
            std::cerr << "rdma_resolve_route failed: " << strerror(errno) << std::endl;
            return -1;
        }
        return wait_for_event(RDMA_CM_EVENT_ROUTE_RESOLVED);
    }

    int setup_connection(rdma_cm_id* id) {
        // 创建保护域（如果尚未创建）
        if (!pd_) {
            pd_ = ibv_alloc_pd(id->verbs);
            if (!pd_) {
                std::cerr << "ibv_alloc_pd failed: " << strerror(errno) << std::endl;
                return -1;
            }
        }

        // 创建完成队列
        ibv_cq* cq = ibv_create_cq(id->verbs, 10, nullptr, nullptr, 0);
        if (!cq) {
            std::cerr << "ibv_create_cq failed: " << strerror(errno) << std::endl;
            return -1;
        }

        // 配置QP属性
        ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.send_cq = cq;
        qp_attr.recv_cq = cq;
        qp_attr.qp_type = IBV_QPT_RC;

        // 创建QP
        if (rdma_create_qp(id, pd_, &qp_attr)) {
            std::cerr << "rdma_create_qp failed: " << strerror(errno) << std::endl;
            ibv_destroy_cq(cq);
            return -1;
        }

        // 保存CQ
        client_cqs_[id] = cq;
        return 0;
    }

    rdma_conn_param* create_conn_param() {
        static rdma_conn_param param;
        memset(&param, 0, sizeof(param));
        param.initiator_depth = 1;
        param.responder_resources = 1;
        param.rnr_retry_count = 7;
        return &param;
    }

    int get_cm_event(rdma_cm_event_type expected, rdma_cm_event** event) {
        if (rdma_get_cm_event(ec_, event)) {
            std::cerr << "rdma_get_cm_event failed: " << strerror(errno) << std::endl;
            return -1;
        }

        if ((*event)->event != expected) {
            std::cerr << "Unexpected event: " << rdma_event_str((*event)->event)
                      << " (expected " << rdma_event_str(expected) << ")" << std::endl;
            rdma_ack_cm_event(*event);
            return -1;
        }
        return 0;
    }

    int wait_for_event(rdma_cm_event_type expected) {
        rdma_cm_event* event;
        if (get_cm_event(expected, &event)) return -1;
        
        rdma_ack_cm_event(event);
        return 0;
    }

    int post_send(rdma_cm_id* id, ibv_mr* mr, size_t size) {
        ibv_sge sge;
        sge.addr = (uintptr_t)mr->addr;
        sge.length = size;
        sge.lkey = mr->lkey;

        ibv_send_wr wr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = (uintptr_t)id;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;

        ibv_send_wr* bad_wr = nullptr;
        return ibv_post_send(id->qp, &wr, &bad_wr);
    }

    int post_recv(rdma_cm_id* id, ibv_mr* mr, size_t size) {
        ibv_sge sge;
        sge.addr = (uintptr_t)mr->addr;
        sge.length = size;
        sge.lkey = mr->lkey;

        ibv_recv_wr wr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = (uintptr_t)id;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        ibv_recv_wr* bad_wr = nullptr;
        return ibv_post_recv(id->qp, &wr, &bad_wr);
    }

    //===============================
    // 成员变量
    //===============================
private:
    std::string ip_;
    std::string port_;
    std::string mode_;

    rdma_event_channel* ec_;
    rdma_cm_id* cm_id_;
    ibv_pd* pd_;

    ibv_mr* local_mr_;      // 本地注册的 MR
    char* local_buffer_;    // 本地缓冲区地址
    size_t buffer_size_;    // 本地缓冲区大小
    bool use_hugepage_;     // 是否使用大页

    std::vector<rdma_cm_id*> client_connections_;           // 服务端保存的所有 client ID
    std::map<rdma_cm_id*, RemoteMemoryInfo> remote_info_;    // 每个 client ID 对应的远程信息
    std::map<rdma_cm_id*, ibv_cq*> client_cqs_;             // 每个 client ID 对应的 CQ
};

#endif // RDMA_ONESIDED_ENDPOINT_H
