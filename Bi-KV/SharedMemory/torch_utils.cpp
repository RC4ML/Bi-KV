#include <torch/extension.h>
#include <torch/torch.h>
#include <cstdint>
torch::Tensor from_blob_wrapper(uintptr_t data_ptr, 
                               const std::vector<int64_t>& shape,
                               torch::ScalarType dtype,
                               const std::string& device_str) {
    void* data = reinterpret_cast<void*>(data_ptr);
    torch::Device device(device_str);
    torch::TensorOptions options = torch::TensorOptions().dtype(dtype).device(device);
    torch::Tensor tensor = torch::from_blob(data, shape, options);
    torch::Tensor copy_tensor = tensor.clone()
    return copy_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("from_blob", &from_blob_wrapper, "Create tensor from memory blob",
          py::arg("data_ptr"), py::arg("shape"),
          py::arg("dtype"), py::arg("device"));
}