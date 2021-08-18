from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bmm_cuda',
    ext_modules=[
        CUDAExtension('bmm_cuda', [
            'bmm_cuda.cpp',
            'bmm_cuda_kernel.cu',],
            extra_compile_args={'nvcc':['-O3','-L/home/ubuntu/magma/magma-2.5.4/lib/  -lmagma', '-I/home/ubuntu/magma/magma-2.5.4/include/' ,'-I/home/ubuntu/magma/magma-2.5.4/control/', '-I/usr/local/cuda/include', '-L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcublas -lcusparse -llapack -lblas -lpthread -lm', '-DADD_'], 
            'gcc':['-O3','-L/home/ubuntu/magma/magma-2.5.4/lib/  -lmagma', '-I/home/ubuntu/magma/magma-2.5.4/include' ,'-I/home/ubuntu/magma/magma-2.5.4/control', '-I/home/ubuntu/magma/magma-2.5.4/magmablas', '-I/usr/local/cuda/include', '-L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcublas -lcusparse -llapack -lblas -lpthread -lm', '-DADD_'],} 
            ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
