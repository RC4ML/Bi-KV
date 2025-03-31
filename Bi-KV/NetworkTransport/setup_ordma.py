from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import torch

# 获取系统路径配置
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
conda_prefix = os.environ.get('CONDA_PREFIX', '/usr')  # 默认回退到系统路径

# 包含目录配置
include_dirs = [
    # PyTorch 头文件
    *torch.utils.cpp_extension.include_paths(),
    
    # CUDA 头文件
    os.path.join(cuda_home, 'include'),
    
    # RDMA 头文件
    os.path.join(conda_prefix, 'include'),         # Conda 环境的头文件
    '/usr/include/infiniband',                     # 系统级 RDMA 头文件
    
    # 其他自定义头文件路径（可选）
    # os.path.join(os.path.dirname(__file__), 'include')
]

# 库目录配置
library_dirs = [
    os.path.join(cuda_home, 'lib64'),              # CUDA 库
    os.path.join(conda_prefix, 'lib'),             # Conda 库
    '/usr/lib',                                    # 系统库
    '/usr/lib/x86_64-linux-gnu'                    # 常见多架构库路径
]

# 需要链接的库
libraries = [
    'rdmacm',     # librdmacm.so
    'ibverbs',    # libibverbs.so
    'cudart',      # CUDA 运行时库
    'torch', 
    'torch_python',
]

# 编译参数配置
extra_compile_args = {
    'cxx': [
        '-std=c++17',        # C++17 标准
        '-O3',               # 最高优化级别
        '-fPIC',            # 位置无关代码
        '-Wall',             # 开启所有警告
        '-Wno-unused',       # 忽略未使用警告
        '-Wno-deprecated'    # 忽略弃用警告
    ]
}

# 定义扩展模块
extension = CppExtension(
    name='rdma_onesided_transport',  # 保持与模块名一致
    sources=[
        'onesided_rdma_module.cpp',  # 主绑定文件
        # 添加其他实现文件（如果有）
        # 'onesided_rdma.cpp'
    ],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args['cxx'],
    language='c++',
)

setup(
    name='rdma_onesided_transport',
    version='0.2',
    description='RDMA One-Sided Transport with CUDA and Verbs support',
    ext_modules=[extension],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

# # 打印关键路径信息用于调试
# print(f"CUDA_HOME: {cuda_home}")
# print(f"CONDA_PREFIX: {conda_prefix}")
# print(f"Include directories: {include_dirs}")
# print(f"Library directories: {library_dirs}\n")