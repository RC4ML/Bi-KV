#ifndef RDMA_ENDPOINT_H
#define RDMA_ENDPOINT_H

#include <arpa/inet.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <torch/extension.h>  

class RDMAEndpoint {
public:
    RDMAEndpoint(const std::string &ip, const std::string &port, const std::string &mode)
        : ip_(ip), port_(port), mode_(mode),
          ec_(nullptr), cm_id_(nullptr),
          pd_(nullptr), cq_(nullptr), qp_(nullptr),
          mr_(nullptr), buffer_(nullptr), buffer_size_(0)
    {}

    ~RDMAEndpoint() {
        cleanup();
    }

    int run_server() {
        if (create_event_channel() != 0) return -1;
        if (create_id() != 0) return -1;
        
        sockaddr_in addr = create_addr(INADDR_ANY);
        if (bind_and_listen(&addr) != 0) return -1;

        rdma_cm_event* event;
        if (get_cm_event(RDMA_CM_EVENT_CONNECT_REQUEST, &event) != 0) return -1;
        
        rdma_cm_id* client_id = event->id;
        rdma_ack_cm_event(event);

        if (create_resources(client_id) != 0) return -1;
        if (accept_connection(client_id) != 0) return -1;

        // 等待连接最终建立
        if (get_cm_event(RDMA_CM_EVENT_ESTABLISHED, &event) != 0) return -1;
        rdma_ack_cm_event(event);
        
        cm_id_ = client_id;  // 更新为实际连接的ID
        std::cout << "Server connection established" << std::endl;
        return 0;
    }

    int connect_client() {
        if (create_event_channel() != 0) return -1;
        if (create_id() != 0) return -1;

        sockaddr_in addr = create_addr();
        if (resolve_address(&addr) != 0) return -1;  // 使用 resolve_address
        if (resolve_route() != 0) return -1;        // 使用 resolve_route

        std::cout<<"address and route resolved"<<std::endl;
        if (create_resources(cm_id_) != 0) return -1;
        if (rdma_connect(cm_id_, create_conn_param()) != 0) {
            perror("rdma_connect");
            return -1;
        }

        rdma_cm_event* event;
        if (get_cm_event(RDMA_CM_EVENT_ESTABLISHED, &event) != 0) return -1;
        rdma_ack_cm_event(event);

        std::cout << "Client connection established" << std::endl;
        return 0;
    }


    // 以下成员函数保持原有实现不变（register_memory, post_receive, post_send, poll_completion等）
    // ...

private:
    // 其他私有函数声明
    int resolve_address(sockaddr_in* addr);
    int resolve_route();

private:
    void cleanup() {
        if (mr_) { ibv_dereg_mr(mr_); mr_ = nullptr; }
        if (buffer_) { free(buffer_); buffer_ = nullptr; }
        if (qp_) { rdma_destroy_qp(cm_id_); qp_ = nullptr; }
        if (cq_) { ibv_destroy_cq(cq_); cq_ = nullptr; }
        if (pd_) { ibv_dealloc_pd(pd_); pd_ = nullptr; }
        if (cm_id_) { rdma_destroy_id(cm_id_); cm_id_ = nullptr; }
        if (ec_) { rdma_destroy_event_channel(ec_); ec_ = nullptr; }
    }

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
        std::cout << "Server listening on port " << port_ << std::endl;
        return 0;
    }

    int get_cm_event(rdma_cm_event_type expected, rdma_cm_event** event) {
        if (rdma_get_cm_event(ec_, event)) {
            perror("rdma_get_cm_event");
            return -1;
        }
        if ((*event)->event != expected) {
            std::cerr << "Expected " << expected << ", got " 
                     << (*event)->event << std::endl;
            rdma_ack_cm_event(*event);
            return -1;
        }
        return 0;
    }

    int create_resources(rdma_cm_id* id) {
        pd_ = ibv_alloc_pd(id->verbs);
        if (!pd_) { perror("ibv_alloc_pd"); return -1; }

        cq_ = ibv_create_cq(id->verbs, 10, nullptr, nullptr, 0);
        if (!cq_) { perror("ibv_create_cq"); return -1; }

        ibv_qp_init_attr qp_attr = create_qp_attr();
        if (rdma_create_qp(id, pd_, &qp_attr)) {
            perror("rdma_create_qp");
            return -1;
        }
        qp_ = id->qp;
        return 0;
    }

    ibv_qp_init_attr create_qp_attr() {
        ibv_qp_init_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.send_cq = cq_;
        attr.recv_cq = cq_;
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
        std::cout << "Server accepted connection" << std::endl;
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

    // 原有成员变量保持不变
    std::string ip_;
    std::string port_;
    std::string mode_;
    rdma_event_channel *ec_;
    rdma_cm_id *cm_id_;
    ibv_pd *pd_;
    ibv_cq *cq_;
    ibv_qp *qp_;
    ibv_mr *mr_;
    char *buffer_;
    size_t buffer_size_;

public:
    // 以下成员函数保持原有实现不变
    int register_memory(size_t size) {
        buffer_size_ = size;
        buffer_ = (char *)malloc(size);
        if (!buffer_) {
            perror("malloc");
            return -1;
        }
        memset(buffer_, 0, size);
        std::cout << "Buffer allocated at: " << (void *)buffer_ << ", size: " << size << " bytes." << std::endl;
        mr_ = ibv_reg_mr(pd_, buffer_, size,
                         IBV_ACCESS_LOCAL_WRITE |
                         IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE);
        if (!mr_) {
            perror("ibv_reg_mr");
            return -1;
        }
        std::cout << "Memory registered, size = " << size << " bytes." << std::endl;
        return 0;
    }

    int post_receive() {
        ibv_sge sge;
        sge.addr   = (uintptr_t)buffer_;
        sge.length = buffer_size_;
        sge.lkey   = mr_->lkey;
        ibv_recv_wr recv_wr;
        memset(&recv_wr, 0, sizeof(recv_wr));
        recv_wr.wr_id   = (uintptr_t)this;
        recv_wr.sg_list = &sge;
        recv_wr.num_sge = 1;
        ibv_recv_wr *bad_wr = nullptr;
        int ret = ibv_post_recv(qp_, &recv_wr, &bad_wr);
        if (ret) { perror("ibv_post_recv"); }
        return ret;
    }
    int post_send(size_t len) {
        // std::cout<<"send data "<<int(buffer_[1])<<std::endl;
        // memset(buffer_, 2, buffer_size_);
        // std::cout<<"send data after memset "<<int(buffer_[1])<<std::endl;

        if (len > buffer_size_) len = buffer_size_;
        ibv_sge sge;
        sge.addr   = (uintptr_t)buffer_;
        sge.length = len;
        sge.lkey   = mr_->lkey;
        ibv_send_wr send_wr;
        memset(&send_wr, 0, sizeof(send_wr));
        send_wr.wr_id      = (uintptr_t)this;
        send_wr.sg_list    = &sge;
        send_wr.num_sge    = 1;
        send_wr.opcode     = IBV_WR_SEND;
        send_wr.send_flags = IBV_SEND_SIGNALED;
        ibv_send_wr *bad_wr = nullptr;
        int ret = ibv_post_send(qp_, &send_wr, &bad_wr);
        if (ret) { perror("ibv_post_send"); }
        return ret;
    }
    int poll_completion() {
        ibv_wc wc;
        int num;
        do {
            num = ibv_poll_cq(cq_, 1, &wc);
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
    char* get_buffer() const { 
        std::cout << "Returning buffer at: " << (void *)buffer_ << ", size: " << buffer_size_ << " bytes." << std::endl;
        return buffer_; 
    }

    torch::Tensor get_buffer_tensor() {
        return torch::from_blob(
            buffer_,
            {static_cast<int64_t>(buffer_size_)},
            torch::TensorOptions()
                .dtype(torch::kUInt8)
                .requires_grad(false));
    }

    size_t get_buffer_size() const { 
        return buffer_size_; 
    }
};


int RDMAEndpoint::resolve_address(sockaddr_in* addr) {
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
        std::cerr << "Expected ADDR_RESOLVED, got " << event->event << std::endl;
        rdma_ack_cm_event(event);
        return -1;
    }

    rdma_ack_cm_event(event);
    return 0;
}

int RDMAEndpoint::resolve_route() {
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
        std::cerr << "Expected ROUTE_RESOLVED, got " << event->event << std::endl;
        rdma_ack_cm_event(event);
        return -1;
    }

    rdma_ack_cm_event(event);
    return 0;
}

#endif // RDMA_ENDPOINT_H