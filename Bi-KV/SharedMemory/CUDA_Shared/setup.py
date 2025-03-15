from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_ipc',
    version='1.0',
    ext_modules=[
        CUDAExtension(
            'ipc_service',
            sources=['ipc_service.cpp'],
            libraries=['rt'],
            extra_compile_args={
                'cxx': ['-O3', '-D_GLIBCXX_USE_CXX11_ABI=1'],
                'nvcc': ['-O3', '--ptxas-options=-v']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)