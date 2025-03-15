from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='torch_utils',
    ext_modules=[
        CppExtension(
            'torch_utils',
            ['torch_utils.cpp'],
            extra_compile_args=['-std=c++17'],
            # 添加以下两行确保 ABI 兼容
            define_macros=[('_GLIBCXX_USE_CXX11_ABI', '1')],  # 与 PyTorch 默认 ABI 一致
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)