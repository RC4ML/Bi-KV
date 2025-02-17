#include "rdma_endpoint.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

PYBIND11_MODULE(rdma_transport, m) {  // 修改模块名称
    m.doc() = "支持多机 RDMA 通信的 C++ 封装模块";

    py::class_<RDMAEndpoint>(m, "RDMAEndpoint")
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             py::arg("ip"), py::arg("port"), py::arg("mode"))
        .def("run_server", &RDMAEndpoint::run_server, "以服务器模式运行")
        .def("connect_client", &RDMAEndpoint::connect_client, "以客户端模式运行")
        .def("register_memory", &RDMAEndpoint::register_memory, 
             "注册内存区域", py::arg("size"))
        .def("post_receive", &RDMAEndpoint::post_receive, "投递接收请求")
        .def("post_send", &RDMAEndpoint::post_send, "投递发送请求", py::arg("len"))
        .def("poll_completion", &RDMAEndpoint::poll_completion, "轮询完成队列")
        .def("get_buffer_tensor", &RDMAEndpoint::get_buffer_tensor,
             py::return_value_policy::reference_internal,  // 关键：确保Tensor与对象生命周期绑定
             "获取buffer的tensor视角")
        .def_property_readonly("buffer_size", &RDMAEndpoint::get_buffer_size);
}