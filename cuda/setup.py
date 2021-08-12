from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bmm_me_cuda',
    ext_modules=[
        CUDAExtension('bmm_me_cuda', [
            'bmm_me_cuda.cpp',
            'bmm_me_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
