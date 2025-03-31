#ifndef RDMA_ENDPOINT_H
#define RDMA_ENDPOINT_H

#include <arpa/inet.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <torch/extension.h>
#include <sys/mman.h>

class RDMAEndpoint {
public:
    RDMAEndpoint(const std::string &ip, const std::string &port, const std::string &mode)
        : ip_(ip), port_(port), mode_(mode), cm_id_(nullptr) {
        if (mode_ == "client") {
            // 客户端模式初始化
        } else if (mode_ == "server") {
            // 服务器模式初始化，例如初始化 rank 到 rdma_cm_id* 的映射
        } else {
            std::cerr << "无效模式: " << mode_ << std::endl;
        }
        pd_ = nullptr;
    }

    ~RDMAEndpoint() {
        cleanup();
    }

    /** 运行服务器，接受多个客户端连接并获取其 rank */
    int run_server(int max_clients) {
        // 创建事件通道和服务器ID
        if (create_event_channel() != 0) return -1;
        if (create_id() != 0) return -1;

        // 配置服务器地址并绑定监听
        sockaddr_in addr = create_addr(INADDR_ANY);
        if (bind_and_listen(&addr) != 0) return -1;

        int connected_clients = 0;
        while (connected_clients < max_clients) {
            // 获取连接请求事件
            rdma_cm_event* event;
            if (get_cm_event(RDMA_CM_EVENT_CONNECT_REQUEST, &event) != 0) break;

            rdma_cm_id* client_id = event->id;
            rdma_ack_cm_event(event);

            // 创建资源并接受连接
            if (create_resources(client_id) != 0) {
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

            // 接收客户端的 rank
            int rank;
            if (receive_rank(client_id, &rank) != 0) {
                rdma_destroy_id(client_id);
                continue;
            }

            // 存储 rank 信息
            client_ranks_[client_id] = rank;
            rank_to_id_[rank] = client_id;
            client_ids_.push_back(client_id);
            connected_clients++;
            // std::cout << "服务器接受了一个客户端连接，rank: " << rank 
            //           << "，总连接数: " << connected_clients << std::endl;

            // // 检查断开连接事件
            // if (get_cm_event(RDMA_CM_EVENT_DISCONNECTED, &event, true) == 0) {
            //     handle_disconnect(event);
            //     rdma_ack_cm_event(event);
            // }
        }

        if (connected_clients == max_clients) {
            // std::cout << "服务器已接受指定数量的客户端: " 
            //           << max_clients << "，停止接受新连接。" << std::endl;
            return 0;
        } else {
            // std::cerr << "服务器未接受足够的客户端连接。" << std::endl;
            return -1;
        }
    }

    /** 客户端连接函数，发送 rank 给服务器 */
    int connect_client(int rank) {
        if (create_event_channel() != 0) return -1;
        if (create_id() != 0) return -1;
        if (!cm_id_) {
            std::cerr << "Error: cm_id_ is nullptr!" << std::endl;
            return -1;
        }
        sockaddr_in addr = create_addr();
        if (resolve_address(&addr) != 0) return -1;
        if (resolve_route() != 0) return -1;
        if (create_resources(cm_id_) != 0) return -1;
        // std::cout << "Client created resources" << std::endl;
        if (rdma_connect(cm_id_, create_conn_param()) != 0) {
            perror("rdma_connect");
            return -1;
        }
        // std::cout << "Client connected server" << std::endl;

        rdma_cm_event* event;
        if (get_cm_event(RDMA_CM_EVENT_ESTABLISHED, &event) != 0) return -1;
        rdma_ack_cm_event(event);

        // 发送 rank 给服务器
        if (send_rank(cm_id_, rank) != 0) {
            std::cerr << "Failed to send rank" << std::endl;
            return -1;
        }

        // std::cout << "Client connection established, rank: " << rank << std::endl;
        return 0;
    }

    /** 为特定 rank 注册内存 */
    int register_memory(int rank, size_t size) {
        if (rank_to_id_.find(rank) == rank_to_id_.end()) {
            std::cerr << "Rank " << rank << " not found." << std::endl;
            return -1;
        }
        rdma_cm_id* id = rank_to_id_[rank];
        return register_memory(id, size);
    }

    /** 为特定 rank 注册大页内存 */
    int register_memory_hugepage(int rank, size_t size) {
        if (rank_to_id_.find(rank) == rank_to_id_.end()) {
            std::cerr << "Rank " << rank << " not found." << std::endl;
            return -1;
        }
        rdma_cm_id* id = rank_to_id_[rank];
        return register_memory_hugepage(id, size);
    }

    int register_memory_client(size_t size) {
        if (mode_ != "client") {
            std::cerr << "此函数仅适用于客户端模式" << std::endl;
            return -1;
        }
        return register_memory(cm_id_, size);
    }
    
    int register_memory_hugepage_client(size_t size) {
        if (mode_ != "client") {
            std::cerr << "此函数仅适用于客户端模式" << std::endl;
            return -1;
        }
        return register_memory_hugepage(cm_id_, size);
    }

    /** 为特定 rank 发布接收操作 */
    int post_receive_by_rank(int rank) {
        if (rank_to_id_.find(rank) == rank_to_id_.end()) {
            std::cerr << "Rank " << rank << " not found." << std::endl;
            return -1;
        }
        rdma_cm_id* id = rank_to_id_[rank];
        return post_receive(id);
    }

    /** 为特定 rank 发布发送操作 */
    int post_send_by_rank(int rank, size_t len) {
        if (rank_to_id_.find(rank) == rank_to_id_.end()) {
            std::cerr << "Rank " << rank << " not found." << std::endl;
            return -1;
        }
        rdma_cm_id* id = rank_to_id_[rank];
        return post_send(id, len);
    }

    /** 为特定 rank 轮询完成事件 */
    int poll_completion_by_rank(int rank) {
        if (rank_to_id_.find(rank) == rank_to_id_.end()) {
            std::cerr << "Rank " << rank << " not found." << std::endl;
            return -1;
        }
        rdma_cm_id* id = rank_to_id_[rank];
        return poll_completion(id);
    }

    int poll_completion() {
        if (mode_ != "client") {
            std::cerr << "此函数仅适用于客户端模式" << std::endl;
            return -1;
        }
        // 调用内部实现，直接使用客户端的 cm_id_
        return poll_completion(cm_id_);
    }

    int post_send(size_t len) {
        if (mode_ != "client") {
            std::cerr << "此函数仅适用于客户端模式" << std::endl;
            return -1;
        }
        return post_send(cm_id_, len);
    }
    
    int post_receive() {
        if (mode_ != "client") {
            std::cerr << "此函数仅适用于客户端模式" << std::endl;
            return -1;
        }
        return post_receive(cm_id_);
    }

    char* get_buffer_by_rank(int rank) const {
        auto it = rank_to_id_.find(rank);
        if (it == rank_to_id_.end()) {
            std::cerr << "Rank " << rank << " not found." << std::endl;
            return nullptr;
        }
        rdma_cm_id* id = it->second; // 使用迭代器访问值
        return get_buffer(id);
    }

    torch::Tensor get_buffer_tensor_by_rank(int rank) {
        if (rank_to_id_.find(rank) == rank_to_id_.end()) {
            std::cerr << "Rank " << rank << " not found." << std::endl;
            return torch::Tensor();
        }
        rdma_cm_id* id = rank_to_id_[rank]; // 非 const 函数，无需修改，但这里逻辑上不一致，建议保持一致性
        return get_buffer_tensor(id);
    }

    size_t get_buffer_size_by_rank(int rank) const {
        auto it = rank_to_id_.find(rank);
        if (it == rank_to_id_.end()) {
            std::cerr << "Rank " << rank << " not found." << std::endl;
            return 0;
        }
        rdma_cm_id* id = it->second; // 使用迭代器访问值
        return get_buffer_size(id);
    }
    char* get_buffer() const {
        if (mode_ != "client") {
            std::cerr << "此函数仅适用于客户端模式" << std::endl;
            return nullptr;
        }
        return get_buffer(cm_id_);
    }
    
    torch::Tensor get_buffer_tensor() {
        if (mode_ != "client") {
            std::cerr << "此函数仅适用于客户端模式" << std::endl;
            return torch::Tensor();
        }
        return get_buffer_tensor(cm_id_);
    }

    size_t get_buffer_size() const {
        if (mode_ != "client") {
            std::cerr << "此函数仅适用于客户端模式" << std::endl;
            return 0;
        }
        return get_buffer_size(cm_id_);
    }



private:
/** 清理所有资源 */
    void cleanup() {
        if (mode_ == "server") {
            // 清理所有客户端连接
            for (auto& [id, rank] : client_ranks_) {
                if (client_mrs_.count(id)) {
                    ibv_dereg_mr(client_mrs_[id]);
                }
                if (client_buffers_.count(id)) {
                    free(client_buffers_[id]);
                }
                if (client_qps_.count(id)) {
                    rdma_destroy_qp(id);
                }
                if (client_cqs_.count(id)) {
                    ibv_destroy_cq(client_cqs_[id]);
                }
                if (id) {
                    rdma_destroy_id(id);
                }
            }
            client_ids_.clear();
            client_cqs_.clear();
            client_qps_.clear();
            client_mrs_.clear();
            client_buffers_.clear();
            client_buffer_sizes_.clear();
            client_ranks_.clear();
            rank_to_id_.clear();
        } else if (mode_ == "client") {
            // 仅清理客户端自身的资源
            if (cm_id_) {
                if (client_mrs_.count(cm_id_)) {
                    ibv_dereg_mr(client_mrs_[cm_id_]);
                }
                if (client_buffers_.count(cm_id_)) {
                    free(client_buffers_[cm_id_]);
                }
                if (client_qps_.count(cm_id_)) {
                    rdma_destroy_qp(cm_id_);
                }
                if (client_cqs_.count(cm_id_)) {
                    ibv_destroy_cq(client_cqs_[cm_id_]);
                }
                rdma_destroy_id(cm_id_);
            }
            client_cqs_.erase(cm_id_);
            client_qps_.erase(cm_id_);
            client_mrs_.erase(cm_id_);
            client_buffers_.erase(cm_id_);
            client_buffer_sizes_.erase(cm_id_);
        }

        // 释放共享的保护域
        if (pd_) {
            ibv_dealloc_pd(pd_);
            pd_ = nullptr;
        }

        // 释放事件通道
        if (ec_) {
            rdma_destroy_event_channel(ec_);
            ec_ = nullptr;
        }

        std::cout << ip_ <<" "<< mode_ <<" RDMAEndpoint cleanup completed." << std::endl;
    }

    /** 处理客户端断开连接 */
    void handle_disconnect(rdma_cm_event* event) {
        rdma_cm_id* id = event->id;
        if (client_ranks_.count(id)) {
            int rank = client_ranks_[id];
            if (client_mrs_.count(id)) {
                ibv_dereg_mr(client_mrs_[id]);
                if (mode_ == "hugepage") {
                    munmap(client_buffers_[id], client_buffer_sizes_[id]);
                } else {
                    free(client_buffers_[id]);
                }
                client_mrs_.erase(id);
                client_buffers_.erase(id);
                client_buffer_sizes_.erase(id);
            }
            if (client_qps_.count(id)) {
                rdma_destroy_qp(id);
                client_qps_.erase(id);
            }
            if (client_cqs_.count(id)) {
                ibv_destroy_cq(client_cqs_[id]);
                client_cqs_.erase(id);
            }
            client_ranks_.erase(id);
            rank_to_id_.erase(rank);
            client_ids_.erase(
                std::remove(client_ids_.begin(), client_ids_.end(), id),
                client_ids_.end());
            rdma_destroy_id(id);
            std::cout << "Client rank " << rank << " disconnected and resources cleaned up." << std::endl;
        }
    }

    /** 接收客户端的 rank */
    int receive_rank(rdma_cm_id* id, int* rank) {
        char* buffer = (char*)malloc(sizeof(int));
        if (!buffer) {
            perror("malloc");
            return -1;
        }
        ibv_mr* mr = ibv_reg_mr(pd_, buffer, sizeof(int), IBV_ACCESS_LOCAL_WRITE);
        if (!mr) {
            perror("ibv_reg_mr");
            free(buffer);
            return -1;
        }

        ibv_sge sge;
        sge.addr = (uintptr_t)buffer;
        sge.length = sizeof(int);
        sge.lkey = mr->lkey;

        ibv_recv_wr recv_wr;
        memset(&recv_wr, 0, sizeof(recv_wr));
        recv_wr.wr_id = (uintptr_t)id;
        recv_wr.sg_list = &sge;
        recv_wr.num_sge = 1;

        ibv_recv_wr* bad_wr = nullptr;
        if (ibv_post_recv(id->qp, &recv_wr, &bad_wr)) {
            perror("ibv_post_recv");
            ibv_dereg_mr(mr);
            free(buffer);
            return -1;
        }

        if (poll_completion(id) != 0) {
            ibv_dereg_mr(mr);
            free(buffer);
            return -1;
        }

        memcpy(rank, buffer, sizeof(int));
        ibv_dereg_mr(mr);
        free(buffer);
        return 0;
    }

    /** 发送 rank 给服务器 */
    int send_rank(rdma_cm_id* id, int rank) {
        char* buffer = (char*)malloc(sizeof(int));
        if (!buffer) {
            perror("malloc");
            return -1;
        }
        memcpy(buffer, &rank, sizeof(int));

        ibv_mr* mr = ibv_reg_mr(pd_, buffer, sizeof(int), IBV_ACCESS_LOCAL_WRITE);
        if (!mr) {
            perror("ibv_reg_mr");
            free(buffer);
            return -1;
        }

        ibv_sge sge;
        sge.addr = (uintptr_t)buffer;
        sge.length = sizeof(int);
        sge.lkey = mr->lkey;

        ibv_send_wr send_wr;
        memset(&send_wr, 0, sizeof(send_wr));
        send_wr.wr_id = (uintptr_t)id;
        send_wr.sg_list = &sge;
        send_wr.num_sge = 1;
        send_wr.opcode = IBV_WR_SEND;
        send_wr.send_flags = IBV_SEND_SIGNALED;

        ibv_send_wr* bad_wr = nullptr;
        if (ibv_post_send(id->qp, &send_wr, &bad_wr)) {
            perror("ibv_post_send");
            ibv_dereg_mr(mr);
            free(buffer);
            return -1;
        }

        if (poll_completion(id) != 0) {
            ibv_dereg_mr(mr);
            free(buffer);
            return -1;
        }

        ibv_dereg_mr(mr);
        free(buffer);
        return 0;
    }

    /** 原有私有函数保持不变，部分摘录 */
    int create_event_channel() {
        ec_ = rdma_create_event_channel();
        if (!ec_) { perror("rdma_create_event_channel"); return -1; }
        return 0;
    }

    int create_id() {
        if (rdma_create_id(ec_, &cm_id_, nullptr, RDMA_PS_TCP)) {
            perror("rdma_create_id");
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
            inet_pton(AF_INET, ip_.c_str(), &sa.sin_addr);
        } else {
            sa.sin_addr.s_addr = addr;
        }
        return sa;
    }

    int bind_and_listen(sockaddr_in* addr) {
        if (rdma_bind_addr(cm_id_, (sockaddr*)addr)) {
            perror("rdma_bind_addr");
            return -1;
        }
        if (rdma_listen(cm_id_, 10)) {
            perror("rdma_listen");
            return -1;
        }
        // std::cout << "Server listening on port " << port_ << std::endl;
        return 0;
    }

    int get_cm_event(rdma_cm_event_type expected, rdma_cm_event** event, bool nonblock = false) {
        if (nonblock) {
            if (rdma_get_cm_event(ec_, event) == -1 && errno == EAGAIN) {
                return -1; // 没有事件
            }
        } else {
            if (rdma_get_cm_event(ec_, event)) {
                perror("rdma_get_cm_event");
                return -1;
            }
        }
        if ((*event)->event != expected) {
            // std::cerr << "Expected " << expected << ", got " 
            //           << (*event)->event << std::endl;
            rdma_ack_cm_event(*event);
            return -1;
        }
        return 0;
    }

    int create_resources(rdma_cm_id* id) {
        if (!pd_) {
            pd_ = ibv_alloc_pd(id->verbs);
            if (!pd_) { perror("ibv_alloc_pd"); return -1; }
        }
        ibv_cq* cq = ibv_create_cq(id->verbs, 10, nullptr, nullptr, 0);
        if (!cq) { perror("ibv_create_cq"); return -1; }
        ibv_qp_init_attr qp_attr = create_qp_attr(cq);
        if (rdma_create_qp(id, pd_, &qp_attr)) {
            perror("rdma_create_qp");
            ibv_destroy_cq(cq);
            return -1;
        }
        client_cqs_[id] = cq;
        client_qps_[id] = id->qp;
        return 0;
    }

    ibv_qp_init_attr create_qp_attr(ibv_cq* cq) {
        ibv_qp_init_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.send_cq = cq;
        attr.recv_cq = cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = 10;
        attr.cap.max_recv_wr = 10;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        return attr;
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

    rdma_conn_param* create_conn_param() {
        static rdma_conn_param param;
        memset(&param, 0, sizeof(param));
        param.initiator_depth = 1;
        param.responder_resources = 1;
        param.rnr_retry_count = 7;
        return &param;
    }

    int resolve_address(sockaddr_in* addr) {
        if (rdma_resolve_addr(cm_id_, nullptr, (sockaddr*)addr, 2000)) {
            perror("rdma_resolve_addr");
            return -1;
        }
        rdma_cm_event* event = nullptr;
        if (rdma_get_cm_event(ec_, &event)) {
            perror("rdma_get_cm_event");
            return -1;
        }
        if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
            // std::cerr << "Expected ADDR_RESOLVED, got " << event->event << std::endl;
            rdma_ack_cm_event(event);
            return -1;
        }
        rdma_ack_cm_event(event);
        return 0;
    }

    int resolve_route() {
        if (rdma_resolve_route(cm_id_, 2000)) {
            perror("rdma_resolve_route");
            return -1;
        }
        rdma_cm_event* event = nullptr;
        if (rdma_get_cm_event(ec_, &event)) {
            perror("rdma_get_cm_event");
            return -1;
        }
        if (event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
            // std::cerr << "Expected ROUTE_RESOLVED, got " << event->event << std::endl;
            rdma_ack_cm_event(event);
            return -1;
        }
        rdma_ack_cm_event(event);
        return 0;
    }

    /** 原有私有函数：为特定客户端注册内存 */
    int register_memory(rdma_cm_id* id, size_t size) {
        char* buffer = (char*)malloc(size);
        if (!buffer) {
            perror("malloc");
            return -1;
        }
        memset(buffer, 0, size);
        ibv_mr* mr = ibv_reg_mr(pd_, buffer, size,
                                IBV_ACCESS_LOCAL_WRITE |
                                IBV_ACCESS_REMOTE_READ |
                                IBV_ACCESS_REMOTE_WRITE);
        if (!mr) {
            perror("ibv_reg_mr");
            free(buffer);
            return -1;
        }
        client_buffers_[id] = buffer;
        client_buffer_sizes_[id] = size;
        client_mrs_[id] = mr;
        // std::cout << "Memory registered for "<<mode_<<", size = " << size << " bytes." << std::endl;
        return 0;
    }

    int register_memory_hugepage(rdma_cm_id* id, size_t size) {
        char* buffer = (char*)mmap(NULL, size, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                                   -1, 0);
        if (buffer == MAP_FAILED) {
            perror("mmap with MAP_HUGETLB failed");
            return -1;
        }
        ibv_mr* mr = ibv_reg_mr(pd_, buffer, size,
                                IBV_ACCESS_LOCAL_WRITE |
                                IBV_ACCESS_REMOTE_READ |
                                IBV_ACCESS_REMOTE_WRITE);
        if (!mr) {
            perror("ibv_reg_mr failed");
            munmap(buffer, size);
            return -1;
        }
        client_buffers_[id] = buffer;
        client_buffer_sizes_[id] = size;
        client_mrs_[id] = mr;
        // std::cout << "Memory registered for client using huge page, size = " << size << " bytes." << std::endl;
        return 0;
    }

    int post_receive(rdma_cm_id* id) {
        char* buffer = client_buffers_[id];
        size_t size = client_buffer_sizes_[id];
        ibv_mr* mr = client_mrs_[id];
        ibv_qp* qp = client_qps_[id];
        ibv_sge sge;
        sge.addr = (uintptr_t)buffer;
        sge.length = size;
        sge.lkey = mr->lkey;
        ibv_recv_wr recv_wr;
        memset(&recv_wr, 0, sizeof(recv_wr));
        recv_wr.wr_id = (uintptr_t)id;
        recv_wr.sg_list = &sge;
        recv_wr.num_sge = 1;
        ibv_recv_wr* bad_wr = nullptr;

        int ret = ibv_post_recv(qp, &recv_wr, &bad_wr);
        if (ret) perror("ibv_post_recv");
        return ret;
    }

    int post_send(rdma_cm_id* id, size_t len) {
        char* buffer = client_buffers_[id];
        size_t size = client_buffer_sizes_[id];
        if (len > size) len = size;
        ibv_mr* mr = client_mrs_[id];
        ibv_qp* qp = client_qps_[id];
        ibv_sge sge;
        sge.addr = (uintptr_t)buffer;
        sge.length = len;
        sge.lkey = mr->lkey;
        ibv_send_wr send_wr;
        memset(&send_wr, 0, sizeof(send_wr));
        send_wr.wr_id = (uintptr_t)id;
        send_wr.sg_list = &sge;
        send_wr.num_sge = 1;
        send_wr.opcode = IBV_WR_SEND;
        send_wr.send_flags = IBV_SEND_SIGNALED;
        ibv_send_wr* bad_wr = nullptr;
        int ret = ibv_post_send(qp, &send_wr, &bad_wr);
        if (ret) perror("ibv_post_send");
        return ret;
    }

    int poll_completion(rdma_cm_id* id) {
        ibv_cq* cq = client_cqs_[id];
        ibv_wc wc;
        int num;
        do {
            num = ibv_poll_cq(cq, 1, &wc);
        } while (num == 0);

        if (num < 0) {
            perror("ibv_poll_cq");
            return -1;
        }
        if (wc.status != IBV_WC_SUCCESS) {
            std::cerr << "Completion error: " << wc.status << std::endl;
            return -1;
        }
        return 0;
    }

    char* get_buffer(rdma_cm_id* id) const {
        auto it = client_buffers_.find(id);
        if (it != client_buffers_.end()) {
            std::cout << "Returning buffer for client, size: " << client_buffer_sizes_.at(id) << " bytes." << std::endl;
            return it->second;
        }
        return nullptr;
    }

    torch::Tensor get_buffer_tensor(rdma_cm_id* id) {
        char* buffer = get_buffer(id);
        size_t size = client_buffer_sizes_.at(id);
        if (buffer) {
            return torch::from_blob(
                buffer,
                {static_cast<int64_t>(size)},
                torch::TensorOptions().dtype(torch::kUInt8).requires_grad(false));
        }
        return torch::Tensor();
    }

    size_t get_buffer_size(rdma_cm_id* id) const {
        return client_buffer_sizes_.at(id);
    }

private:
    std::string ip_;
    std::string port_;
    std::string mode_;
    rdma_event_channel* ec_;
    rdma_cm_id* cm_id_;  // 服务器监听的ID
    ibv_pd* pd_;         // 共享的保护域

    // 用于管理多个客户端的容器
    std::vector<rdma_cm_id*> client_ids_;                // 客户端ID列表
    std::map<rdma_cm_id*, ibv_cq*> client_cqs_;          // 每个客户端的CQ
    std::map<rdma_cm_id*, ibv_qp*> client_qps_;          // 每个客户端的QP
    std::map<rdma_cm_id*, ibv_mr*> client_mrs_;          // 每个客户端的内存区域
    std::map<rdma_cm_id*, char*> client_buffers_;        // 每个客户端的缓冲区
    std::map<rdma_cm_id*, size_t> client_buffer_sizes_;  // 每个客户端的缓冲区大小
    std::map<rdma_cm_id*, int> client_ranks_;            // 每个客户端的 rank
    std::map<int, rdma_cm_id*> rank_to_id_;              // rank 到 ID 的映射
};

#endif // RDMA_ENDPOINT_H