from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bmm_cuda',
    ext_modules=[
        CUDAExtension('bmm_cuda', [
            'bmm_cuda.cpp',
            'bmm_cuda_kernel.cu',],
            extra_compile_args={'nvcc':['-O3', '-DADD_ -Xcompiler "-fPIC -Wall -Wno-unused-function" -std=c++11  -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75', '-L/home/ubuntu/magma/magma-2.5.4/lib/  -lmagma_sparse -lmagma', '-I/home/ubuntu/magma/magma-2.5.4/include/' , '-I/usr/local/cuda/include', '-L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcublas -lcusparse -llapack -lblas -lpthread -lm', '-DADD_'], 
            'gcc':['-O3','-L/home/ubuntu/magma/magma-2.5.4/lib/ -lmagma_sparse -lmagma', '-I/home/ubuntu/magma/magma-2.5.4/include' , '-I/usr/local/cuda/include', '-L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcublas -lcusparse -llapack -lblas -lpthread -lm', '-DADD_'],} 
            ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
