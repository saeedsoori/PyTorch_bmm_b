import math
from torch import nn
from torch.autograd import Function
import torch
import ctypes

import bmm_cuda

torch.manual_seed(42)


class BMM():
    def __init__(self, A, B, C, batch_size, offset_A, offset_B, offset_C):

        foo = bmm_cuda.Foo()
        foo.set_pointers(A, B, C, batch_size, offset_A, offset_B, offset_C)
        # print(foo.getKey())
        self.x = 1

    def forward(A, B, C, m , n, k, batch_size, offset_A, offset_B, offset_C):
        # print('running....')
        # m_arr = (ctypes.c_int * len(m))(*m)
        # n_arr = (ctypes.c_int * len(n))(*n)
        # k_arr = (ctypes.c_int * len(k))(*k)
        # print('converting finshed....')
        # return bmm_cuda.forward(A, B, m_arr, n_arr, k_arr)

        # return bmm_cuda.forward(A, B, C, m, n, k, batch_size, offset_A, offset_B, offset_C)
        return self.foo.forward(A, B, C, m, n, k, batch_size, offset_A, offset_B, offset_C)

    # def cublas_forward(A, B, C, m , n, k, batch_size, offset_A, offset_B, offset_C):
        # print('running single mode....')
        # m_arr = (ctypes.c_int * len(m))(*m)
        # n_arr = (ctypes.c_int * len(n))(*n)
        # k_arr = (ctypes.c_int * len(k))(*k)
        # print('converting finshed....')
        # return bmm_cuda.forward(A, B, m_arr, n_arr, k_arr)
        # return bmm_cuda.cublas_gemm_call(A, B, C, m , n, k, batch_size, offset_A, offset_B, offset_C)
