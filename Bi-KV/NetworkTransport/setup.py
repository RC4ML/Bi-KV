from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# 配置头文件路径和库路径
extra_compile_args = ['-O3', '-Wall', '-std=c++17', '-fPIC']
libraries = ['rdmacm', 'ibverbs', 'torch', 'torch_python']
include_dirs = [
    os.path.join(os.environ['CONDA_PREFIX'], 'include'),  # 根据你的环境调整
    '/usr/include/infiniband'
]

setup(
    name='rdma_transport',  # 修改模块名称
    version='0.1',
    ext_modules=[
        CppExtension(
            'rdma_transport',  # 修改模块名称
            sources=['rdma_module.cpp'],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            libraries=libraries,
            language='c++'
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
)