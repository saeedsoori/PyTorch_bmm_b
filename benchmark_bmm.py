from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=4)
parser.add_argument('-f', '--features', type=int, default=32)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
parser.add_argument('-n', '--n', type=int, default=2)

options = parser.parse_args()

# change this line
from cuda.bmm import BMM



options.cuda = True
device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

# generate "n" random matrix with different #columns
# r_size = [32, 64, 128, 198, 256]
r_size = [2,4,8,16]
A = []
B = []
C = []
C_true = []
mshapes = []
nshapes = []
kshapes = []
index = torch.randint(0, len(r_size), (options.n,))

pytorch_min = math.inf
pytorch_time = 0
magma_min = math.inf
magma_time = 0

for i in range(options.n):
    A_s = torch.randn(options.batch_size, r_size[index[i]], **kwargs)
    B_s = torch.randn(r_size[index[i]], r_size[index[i]] + 32, **kwargs)
    C_s = torch.zeros(options.batch_size, r_size[index[i]] + 32, **kwargs)
    start = time.time()
    C_s_true = torch.matmul(A_s, B_s)
    elapsed = time.time() - start
    pytorch_min = min(pytorch_min, elapsed)
    pytorch_time += elapsed

    A.append(A_s)
    B.append(B_s)
    C.append(C_s)
    C_true.append(C_s_true)
    mshapes.append(A_s.shape[0])
    nshapes.append(B_s.shape[1])
    kshapes.append(A_s.shape[1])
    # print(A[i].shape)
    # print(B[i].shape)
    print('*'*10)

Mul = BMM()
m_arr = torch.cuda.IntTensor(mshapes)
n_arr = torch.cuda.IntTensor(nshapes)
k_arr = torch.cuda.IntTensor(kshapes)

start = time.time()
result = BMM.forward(A, B, C, m_arr, n_arr, k_arr, options.n)
elapsed = time.time() - start
magma_min = min(magma_min, elapsed)
magma_time += elapsed

# 
# print('results...........')
# print('A tensors:', A)
# print('B tensors:', B)
# print('C tensors:', C)
# print('C true tensors:', C_true)

for i in range(options.n):
    print(torch.allclose(C[i], C_true[i]))
    
scale = TIME_SCALES[options.scale]
pytorch_min *= scale
magma_min *= scale
pytorch_average = pytorch_time / 1 * scale
magma_average = magma_time / 1 * scale

print('PyTorch: {0:.3f}/{1:.3f} {4} | Magma {2:.3f}/{3:.3f} {4}'.format(
    pytorch_min, pytorch_average, magma_min, magma_average,
    options.scale))


# result_single = BMM.single(A_s, B_s, C_s, A_s.shape[0], B_s.shape[1], A_s.shape[1])



# print('Single mode: C true:', C_s_true)
# print('Single mode: C magma:', C_s)



# Force CUDA initialization
# new_h, new_C = rnn(X, (h, C))
# (new_h.sum() + new_C.sum()).backward()


# for _ in range(options.runs):
#     rnn.zero_grad()

    # start = time.time()
#     new_h, new_C = rnn(X, (h, C))
    # elapsed = time.time() - start
#     forward_min = min(forward_min, elapsed)
#     forward_time += elapsed

#     start = time.time()
#     (new_h.sum() + new_C.sum()).backward()
#     elapsed = time.time() - start
#     backward_min = min(backward_min, elapsed)
#     backward_time += elapsed


