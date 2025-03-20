#include "rdma_endpoint.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(rdma_transport, m) {
    py::class_<RDMAEndpoint>(m, "RDMAEndpoint")
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             py::arg("ip"), py::arg("port"), py::arg("mode"),
             "初始化 RDMAEndpoint。\n"
             "参数:\n"
             "  ip (str): 服务器 IP 地址。\n"
             "  port (str): 端口号。\n"
             "  mode (str): 'server' 或 'client' 或 'hugepage'（服务器使用大页）。")

        // 服务器相关函数
        .def("run_server", &RDMAEndpoint::run_server,
             py::arg("max_clients"),
             "以服务器模式运行并接受指定数量的客户端连接。\n"
             "参数:\n"
             "  max_clients (int): 最大客户端连接数。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")

        // 客户端相关函数
        .def("connect_client", &RDMAEndpoint::connect_client,
             py::arg("rank"),
             "以客户端模式连接到服务器并发送 rank。\n"
             "参数:\n"
             "  rank (int): 客户端的 rank。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")

        // 内存注册接口
        .def("register_memory", static_cast<int (RDMAEndpoint::*)(int, size_t)>(&RDMAEndpoint::register_memory),
             py::arg("rank"), py::arg("size"),
             "为指定 rank 的客户端注册普通内存区域（服务器模式）。\n"
             "参数:\n"
             "  rank (int): 客户端的 rank。\n"
             "  size (int): 内存大小（字节）。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")
        .def("register_memory_client", static_cast<int (RDMAEndpoint::*)(size_t)>(&RDMAEndpoint::register_memory_client),
             py::arg("size"),
             "为客户端自身注册普通内存区域（客户端模式）。\n"
             "参数:\n"
             "  size (int): 内存大小（字节）。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")

        // RDMA 操作接口
        .def("post_receive_by_rank", &RDMAEndpoint::post_receive_by_rank,
             py::arg("rank"),
             "为指定 rank 的客户端投递接收请求（服务器模式）。\n"
             "参数:\n"
             "  rank (int): 客户端的 rank。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")
        .def("post_receive", static_cast<int (RDMAEndpoint::*)()>(&RDMAEndpoint::post_receive),
             "为客户端自身投递接收请求（客户端模式）。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")

        .def("post_send_by_rank", &RDMAEndpoint::post_send_by_rank,
             py::arg("rank"), py::arg("len"),
             "为指定 rank 的客户端投递发送请求（服务器模式）。\n"
             "参数:\n"
             "  rank (int): 客户端的 rank。\n"
             "  len (int): 发送数据长度（字节）。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")
        .def("post_send", static_cast<int (RDMAEndpoint::*)(size_t)>(&RDMAEndpoint::post_send),
             py::arg("len"),
             "为客户端自身投递发送请求（客户端模式）。\n"
             "参数:\n"
             "  len (int): 发送数据长度（字节）。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")

        .def("poll_completion_by_rank", &RDMAEndpoint::poll_completion_by_rank,
             py::arg("rank"),
             "为指定 rank 的客户端轮询完成队列（服务器模式）。\n"
             "参数:\n"
             "  rank (int): 客户端的 rank。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")
        .def("poll_completion", static_cast<int (RDMAEndpoint::*)()>(&RDMAEndpoint::poll_completion),
             "为客户端自身轮询完成队列（客户端模式）。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")

        // 缓冲区访问接口
        .def("get_buffer_by_rank", &RDMAEndpoint::get_buffer_by_rank,
             py::arg("rank"),
             "获取指定 rank 的客户端缓冲区（服务器模式）。\n"
             "参数:\n"
             "  rank (int): 客户端的 rank。\n"
             "返回:\n"
             "  bytes: 缓冲区数据，若失败则返回 None。")
        .def("get_buffer", static_cast<char* (RDMAEndpoint::*)() const>(&RDMAEndpoint::get_buffer),
             "获取客户端自身的缓冲区（客户端模式）。\n"
             "返回:\n"
             "  bytes: 缓冲区数据，若失败则返回 None。")

        .def("get_buffer_tensor_by_rank", &RDMAEndpoint::get_buffer_tensor_by_rank,
             py::arg("rank"),
             "将指定 rank 的客户端缓冲区转换为 PyTorch Tensor（服务器模式）。\n"
             "参数:\n"
             "  rank (int): 客户端的 rank。\n"
             "返回:\n"
             "  torch.Tensor: 缓冲区的 Tensor 表示，若失败则返回空 Tensor。")
        .def("get_buffer_tensor", static_cast<torch::Tensor (RDMAEndpoint::*)()>(&RDMAEndpoint::get_buffer_tensor),
             "将客户端自身缓冲区转换为 PyTorch Tensor（客户端模式）。\n"
             "返回:\n"
             "  torch.Tensor: 缓冲区的 Tensor 表示，若失败则返回空 Tensor。")

        .def("get_buffer_size_by_rank", &RDMAEndpoint::get_buffer_size_by_rank,
             py::arg("rank"),
             "获取指定 rank 的客户端缓冲区大小（服务器模式）。\n"
             "参数:\n"
             "  rank (int): 客户端的 rank。\n"
             "返回:\n"
             "  int: 缓冲区大小（字节），若失败则返回 0。")
        .def("get_buffer_size", static_cast<size_t (RDMAEndpoint::*)() const>(&RDMAEndpoint::get_buffer_size),
             "获取客户端自身缓冲区大小（客户端模式）。\n"
             "返回:\n"
             "  int: 缓冲区大小（字节），若失败则返回 0。");
}
