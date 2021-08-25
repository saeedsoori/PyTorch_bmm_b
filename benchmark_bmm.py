from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-f', '--features', type=int, default=32)
parser.add_argument('-r', '--runs', type=int, default=10)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
parser.add_argument('-n', '--n', type=int, default=1)

options = parser.parse_args()

# change this line
from cuda.bmm import BMM



options.cuda = True
device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

# generate "n" random matrix with different #columns
# r_size = [16, 24, 32, 64, 72, 128]
# r_size = [2,4,8,16]
r_size = [32]
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

sum_size_A = 0
sum_size_B = 0
sum_size_C = 0

for i in range(options.n):
    A_s = torch.randn(options.batch_size, r_size[index[i]], **kwargs)
    B_s = torch.randn(r_size[index[i]], r_size[index[i]] + 3, **kwargs)
    C_s = torch.zeros(options.batch_size, r_size[index[i]] + 3, **kwargs)

    print(A_s)
    print(B_s)
    print(C_s)
    # Force CUDA initialization
    C_s_true = torch.matmul(A_s, B_s)

    A.append(A_s)
    B.append(B_s)
    C.append(C_s)
    
    mshapes.append(A_s.shape[0])
    nshapes.append(B_s.shape[1])
    kshapes.append(A_s.shape[1])

    sum_size_A = sum_size_A + A_s.numel()
    sum_size_B = sum_size_B + B_s.numel()
    sum_size_C = sum_size_C + C_s.numel()

# adding one extra elements since magma needs it
mshapes.append(0)
nshapes.append(0)
kshapes.append(0)

### making a contiguous tensor
A_con = torch.zeros(sum_size_A, **kwargs)
B_con = torch.zeros(sum_size_B, **kwargs)
C_con = torch.zeros(sum_size_C, **kwargs)

# print('tensors created with size:', sum_size_A)
offset_A = 0
offset_B = 0
offset_C = 0
all_offset_A = [offset_A]
all_offset_B= [offset_B]
all_offset_C = [offset_C]
for i in range(options.n):
    A_con[0 + offset_A:A[i].numel() + offset_A] = torch.reshape(A[i], [1,-1])
    B_con[0 + offset_B:B[i].numel() + offset_B] = torch.reshape(B[i], [1,-1])
    C_con[0 + offset_C:C[i].numel() + offset_C] = torch.reshape(C[i], [1,-1])
    offset_A = offset_A + A[i].numel()
    offset_B = offset_B + B[i].numel()
    offset_C = offset_C + C[i].numel()
    all_offset_A.append(offset_A)
    all_offset_B.append(offset_B)
    all_offset_C.append(offset_C)



m_arr = torch.cuda.IntTensor(mshapes)
n_arr = torch.cuda.IntTensor(nshapes)
k_arr = torch.cuda.IntTensor(kshapes)


Mul = BMM(A_con, B_con, C_con, options.n, all_offset_A, all_offset_B, all_offset_C)
C_con = torch.zeros(sum_size_C, **kwargs)
# result = BMM.forward(A, B, C, m_arr, n_arr, k_arr, options.n)
# result = Mul.forward(A_con, B_con, C_con, m_arr, n_arr, k_arr, options.n, all_offset_A, all_offset_B, all_offset_C)
result = Mul.Cublasforward(A_con, B_con, C_con, m_arr, n_arr, k_arr, options.n, all_offset_A, all_offset_B, all_offset_C)
# C_con = 
for j in range(options.runs):
    C_true = []
    for i in range(options.n):
        start = time.time()
        C_s_true = torch.matmul(A[i], B[i])
        elapsed = time.time() - start
        pytorch_min = min(pytorch_min, elapsed)
        pytorch_time += elapsed
        C_true.append(C_s_true)

    # calling magma
    # print('start calling mamgma')
    start = time.time()
    # result = Mul.forward(A_con, B_con, C_con, m_arr, n_arr, k_arr, options.n, all_offset_A, all_offset_B, all_offset_C)
    result = Mul.Cublasforward(A_con, B_con, C_con, m_arr, n_arr, k_arr, options.n, all_offset_A, all_offset_B, all_offset_C)
    elapsed = time.time() - start
    magma_min = min(magma_min, elapsed)
    magma_time += elapsed
    C_con = torch.zeros(sum_size_C, **kwargs)

    # for k in range(options.n):
    #     C[k] = C_con[0 + all_offset_C[k]: C_true[k].numel() + all_offset_C[k]]
    #     if not torch.allclose(C[k].view_as(C_true[k]), C_true[k]):
    #         print(C[k].view_as(C_true[k])-C_true[k])
            # print('#'*20)
        
    

 
scale = TIME_SCALES[options.scale]
pytorch_min *= scale
magma_min *= scale
pytorch_average = pytorch_time / options.runs * scale
magma_average = magma_time / options.runs * scale

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


