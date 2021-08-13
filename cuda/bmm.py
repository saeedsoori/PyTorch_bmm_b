import math
from torch import nn
from torch.autograd import Function
import torch

import bmm_cuda

torch.manual_seed(42)


class BMM():
    def __init__(self):
        self.x = 1

    def forward(A, B):
        return bmm_cuda.forward(A, B)
