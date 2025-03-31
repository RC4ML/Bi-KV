#include "onesided_rdma.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(rdma_onesided_transport, m) {
    py::class_<RDMAOneSidedEndpoint>(m, "RDMAOneSidedEndpoint")
        // 构造函数
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             py::arg("ip"), py::arg("port"), py::arg("mode"),
             "初始化 RDMAOneSidedEndpoint。\n"
             "参数:\n"
             "  ip (str): 服务器 IP 地址。\n"
             "  port (str): 端口号。\n"
             "  mode (str): 'server' 或 'client'。\n")

        // 服务器相关函数
        .def("run_server", &RDMAOneSidedEndpoint::run_server,
             py::arg("max_clients"), py::arg("local_cpu_size"), py::arg("local_gpu_size"), py::arg("hugepage") = false,
             "以服务器模式运行并接受指定数量的客户端连接。\n"
             "参数:\n"
             "  max_clients (int): 最大客户端连接数。\n"
             "  local_cpu_size (int): 服务器本地 CPU 内存大小（字节）。\n"
             "  local_gpu_size (int): 服务器本地 GPU 内存大小（字节）。\n"
             "  hugepage (bool): 是否使用大页内存（默认 False）。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")

        // 客户端相关函数
        .def("connect_client", &RDMAOneSidedEndpoint::connect_client,
             py::arg("rank"), py::arg("cpu_size"), py::arg("gpu_size"), py::arg("hugepage") = false,
             "以客户端模式连接到服务器并发送 rank。\n"
             "参数:\n"
             "  rank (int): 客户端的 rank ID。\n"
             "  cpu_size (int): 客户端 CPU 内存大小（字节）。\n"
             "  gpu_size (int): 客户端 GPU 内存大小（字节）。\n"
             "  hugepage (bool): 是否使用大页内存（默认 False）。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")

        // RDMA 操作接口（服务器模式）
        .def("post_rdma_write", &RDMAOneSidedEndpoint::post_rdma_write,
             py::arg("rank"), py::arg("size"), py::arg("src_type"), py::arg("dst_type"),
             py::arg("local_offset") = 0, py::arg("remote_offset") = 0,
             "从服务器本地内存写入到客户端内存（服务器模式）。\n"
             "参数:\n"
             "  rank (int): 客户端的 rank ID。\n"
             "  size (int): 数据大小（字节）。\n"
             "  src_type (str): 源内存类型 ('cpu' 或 'gpu')。\n"
             "  dst_type (str): 目标内存类型 ('cpu' 或 'gpu')。\n"
             "  local_offset (int): 本地内存偏移量（字节，默认 0）。\n"
             "  remote_offset (int): 远程内存偏移量（字节，默认 0）。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")

        .def("post_rdma_read", &RDMAOneSidedEndpoint::post_rdma_read,
             py::arg("rank"), py::arg("size"), py::arg("src_type"), py::arg("dst_type"),
             py::arg("local_offset") = 0, py::arg("remote_offset") = 0,
             "从客户端内存读取到服务器本地内存（服务器模式）。\n"
             "参数:\n"
             "  rank (int): 客户端的 rank ID。\n"
             "  size (int): 数据大小（字节）。\n"
             "  src_type (str): 源内存类型 ('cpu' 或 'gpu')。\n"
             "  dst_type (str): 目标内存类型 ('cpu' 或 'gpu')。\n"
             "  local_offset (int): 本地内存偏移量（字节，默认 0）。\n"
             "  remote_offset (int): 远程内存偏移量（字节，默认 0）。\n"
             "返回:\n"
             "  int: 0 表示成功，-1 表示失败。")

        .def("get_server_cpu_tensor", &RDMAOneSidedEndpoint::get_server_cpu_tensor,
             py::arg("rank"),
             "获取指定客户端 rank 对应的服务器 CPU 内存的 Tensor。")

        .def("get_server_gpu_tensor", &RDMAOneSidedEndpoint::get_server_gpu_tensor,
             py::arg("rank"),
             "获取指定客户端 rank 对应的服务器 GPU 内存的 Tensor。")

        .def("get_client_cpu_tensor", &RDMAOneSidedEndpoint::get_client_cpu_tensor,
             "获取客户端 CPU 内存的 Tensor。")

        .def("get_client_gpu_tensor", &RDMAOneSidedEndpoint::get_client_gpu_tensor,
             "获取客户端 GPU 内存的 Tensor.");
}
