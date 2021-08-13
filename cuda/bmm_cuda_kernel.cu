#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <string.h> //for memcpy

#include <magma_v2.h>

namespace {


// template <typename scalar_t>
// __global__ void bmm_cuda_forward_kernel(
//     const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gates,
//     const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_cell,
//     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_h,
//     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_cell,
//     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input_gate,
//     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output_gate,
//     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> candidate_cell) {
//   //batch index
//   const int n = blockIdx.y;
//   // column index
//   const int c = blockIdx.x * blockDim.x + threadIdx.x;
//   if (c < gates.size(2)){
//     input_gate[n][c] = sigmoid(gates[n][0][c]);
//     output_gate[n][c] = sigmoid(gates[n][1][c]);
//     candidate_cell[n][c] = elu(gates[n][2][c]);
//     new_cell[n][c] =
//         old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
//     new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
//   }
// }


} // namespace

int bmm_cuda_forward(
    double* pA,
    double* pB,
    int m,
    int n,
    int k) {
  // auto X = torch::cat({old_h, input}, /*dim=*/1);
  // auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

  // const auto batch_size = old_cell.size(0);
  // const auto state_size = old_cell.size(1);

  // auto gates = gate_weights.reshape({batch_size, 3, state_size});
  // auto new_h = torch::zeros_like(old_cell);
  // auto new_cell = torch::zeros_like(old_cell);
  // auto input_gate = torch::zeros_like(old_cell);
  // auto output_gate = torch::zeros_like(old_cell);
  // auto candidate_cell = torch::zeros_like(old_cell);

  // const int threads = 1024;
  // const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  // AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
  //   lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
  //       gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
  //       old_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
  //       new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
  //       new_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
  //       input_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
  //       output_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
  //       candidate_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  // }));
  magma_trans_t transA = MagmaNoTrans;
  magma_trans_t transB = MagmaNoTrans;

  // magmablas_dgemm_vbatched(       transA,
  //     /* magma_trans_t */         transB,
  //     /* magma_int_t * */         d_m,
  //     /* magma_int_t * */         d_n,
  //     /* magma_int_t * */         d_k,
  //     /* double */                alpha,
  //     /* double const *const * */ dA_array,
  //     /* magma_int_t * */         d_ldda,
  //     /* double const *const * */ dB_array,
  //     /* magma_int_t * */         d_lddb,
  //     /* double */                beta,
  //     /* double ** */             dC_array,
  //     /* magma_int_t * */         d_lddc,
  //     /* magma_int_t */           batchCount,
  //     /* magma_queue_t */         queue);

  return m;
}
