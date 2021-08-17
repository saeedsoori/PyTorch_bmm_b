from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bmm_cuda',
    ext_modules=[
        CUDAExtension('bmm_cuda', [
            'bmm_cuda.cpp',
            'bmm_cuda_kernel.cu',],
            extra_compile_args=['-I/content/magma-2.5.4/include/ -L. lmagma'],
            ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
