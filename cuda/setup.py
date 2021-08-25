from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bmm_cuda',
    ext_modules=[
        CUDAExtension('bmm_cuda', [
            'bmm_cuda.cpp',],
            extra_compile_args={'cxx':['-O3', '-I/usr/local/cuda/include', '-L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcublas -lcusparse -llapack -lblas -lpthread -lm'],} 
            ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
