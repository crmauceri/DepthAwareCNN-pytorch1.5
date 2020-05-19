from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='depthavgpooling',
    ext_modules=[
        CUDAExtension('depthavgpooling', [
            'src/depthavgpooling_cuda.c',
            'src/depthavgpooling_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })