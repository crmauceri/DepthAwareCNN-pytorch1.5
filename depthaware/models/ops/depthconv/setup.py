from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='depthconv',
    ext_modules=[
        CUDAExtension('depthconv', [
            'src/depthconv_cuda.cpp',
            'src/depthconv_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })