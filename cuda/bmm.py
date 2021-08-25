import math
from torch import nn
from torch.autograd import Function
import torch
import ctypes

import bmm_cuda

torch.manual_seed(42)


class BMM():
    def __init__(self, A, B, C, batch_size, offset_A, offset_B, offset_C):

        self.mul_op = bmm_cuda.cublas_class()
        # self.foo.set_pointers(A, B, C, batch_size, offset_A, offset_B, offset_C)
        # self.foo.set_pointers_cublas(A, B, C, batch_size, offset_A, offset_B, offset_C)
        # print(foo.getKey())
        self.x = 1

    def forward(self, A, B, C, m , n, k, batch_size, offset_A, offset_B, offset_C):
        return self.mul_op.fooforward(A, B, C, m, n, k, batch_size, offset_A, offset_B, offset_C)

    def Cublasforward(self, A, B, C, m , n, k, batch_size, offset_A, offset_B, offset_C):
        return self.mul_op.Cublasforward(A, B, C, m, n, k, batch_size, offset_A, offset_B, offset_C)

    