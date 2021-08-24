#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
// #include "testings.h"

#include <string.h> //for memcpy

// #include <magma_v2.h>

// Pulled from magma test code
#define TESTING_CHECK( err )                                                 \
    do {                                                                     \
        magma_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
                     #err, __FILE__, __LINE__,                               \
                     (long long) err_, magma_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )

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
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int* m,
    int* n,
    int* k,
    int batch_size,
    std::vector<int> offset_A,
    std::vector<int> offset_B,
    std::vector<int> offset_C) {
  

  float ** hA_array;
  float ** hB_array;
  float ** hC_array;

  float const* * dA_array;
  float const* * dB_array;
  float **dC_array;

  magma_int_t* d_m;
  magma_int_t* d_n;
  magma_int_t* d_k;

  double  alpha = 1.0;
  double  beta = 0.0;
  magma_int_t* d_lddb;
  magma_int_t* d_ldda;
  magma_int_t* d_lddc;



  magma_trans_t transA = MagmaNoTrans;
  magma_trans_t transB = MagmaNoTrans;

  magma_int_t batchCount = batch_size;
  magma_queue_t queue;
  magma_device_t device;
  // std::cout<<"initialization finsihed..."<<"\n";

  magma_getdevice( &device );
  magma_queue_create( device, &queue );


  // check if 
  // std::cout<<"is the input correct?"<<"\n";
  // int *m_dst, *n_dst, *k_dst;
  // TESTING_CHECK( magma_malloc_cpu( (void**)&m_dst, sizeof(int*)*batchCount ) );
  // TESTING_CHECK( magma_malloc_cpu( (void**)&n_dst, sizeof(int*)*batchCount ) );
  // TESTING_CHECK( magma_malloc_cpu( (void**)&k_dst, sizeof(int*)*batchCount ) );
  // int nelem = batchCount;
  // magma_getvector(nelem, sizeof(int), m, 1, m_dst, 1, queue); 
  // magma_getvector(nelem, sizeof(int), n, 1, n_dst, 1, queue); 
  // magma_getvector(nelem, sizeof(int), k, 1, k_dst, 1, queue); 
  // std::cout<<"checking for m is finsihed: "<<m_dst[0]<<" "<<m_dst[1]<<"\n";
  // std::cout<<"checking for n is finsihed: "<<n_dst[0]<<" "<<n_dst[1]<<"\n";
  // std::cout<<"checking for k is finsihed: "<<k_dst[0]<<" "<<k_dst[1]<<"\n";


  TESTING_CHECK( magma_malloc_cpu( (void**)&hA_array, sizeof(float*)*batchCount ) );
  TESTING_CHECK( magma_malloc_cpu( (void**)&hB_array, sizeof(float*)*batchCount ) );
  TESTING_CHECK( magma_malloc_cpu( (void**)&hC_array, sizeof(float*)*batchCount ) );

  // TESTING_CHECK( magma_malloc((void**)&d_m, (batchCount+1)*sizeof(magma_int_t)) );
  // TESTING_CHECK( magma_malloc((void**)&d_n, (batchCount+1)*sizeof(magma_int_t)) );
  // TESTING_CHECK( magma_malloc((void**)&d_k, (batchCount+1)*sizeof(magma_int_t)) );

  // TESTING_CHECK( magma_malloc((void**)&d_ldda, (batchCount+1)*sizeof(magma_int_t) ) );
  // TESTING_CHECK( magma_malloc((void**)&d_lddb, (batchCount+1)*sizeof(magma_int_t) ) );
  // TESTING_CHECK( magma_malloc((void**)&d_lddc, (batchCount+1)*sizeof(magma_int_t) ) );

  for (int i = 0; i < batchCount; ++i)
  {
    // std::cout<<"processing input tensor:"<< i<< " \n";

    hA_array[i] = (float *) A.data_ptr() + offset_A[i];
    hB_array[i] = (float *) B.data_ptr() + offset_B[i];
    hC_array[i] = (float *) C.data_ptr() + offset_C[i];
  }



  // dA_array is the array of pointers need by dgemm
  // d_A_elems are the actual mtx elements being pointed to
  // hA_array is the host side pointers that will get passed to dA_array
  // double const* * dA_array;
  // double const* * dB_array;
  // double ** dC_array;

  TESTING_CHECK( magma_malloc( (void**)&dA_array, sizeof(float*)*batchCount ) );
  TESTING_CHECK( magma_malloc( (void**)&dB_array, sizeof(float*)*batchCount ) );
  TESTING_CHECK( magma_malloc( (void**)&dC_array, sizeof(float*)*batchCount ) );

  magma_setvector(batchCount, sizeof(float*), hA_array, 1, dA_array, 1, queue);
  magma_setvector(batchCount, sizeof(float*), hB_array, 1, dB_array, 1, queue);
  magma_setvector(batchCount, sizeof(float*), hC_array, 1, dC_array, 1, queue);

  // std::cout<<"moving host array to device finsihed..."<<"\n";
  // magma_setvector(batchCount, sizeof(magma_int_t), n_dst, 1, d_m, 1, queue);
  // magma_setvector(batchCount, sizeof(magma_int_t), m_dst, 1, d_n, 1, queue);
  // magma_setvector(batchCount, sizeof(magma_int_t), k_dst, 1, d_k, 1, queue);
  // magma_setvector(batchCount, sizeof(magma_int_t), n_dst, 1, d_lddb, 1, queue);
  // magma_setvector(batchCount, sizeof(magma_int_t), k_dst, 1, d_ldda, 1, queue);
  // magma_setvector(batchCount, sizeof(magma_int_t), n_dst, 1, d_lddc, 1, queue);

  // magma_setvector(batchCount, sizeof(magma_int_t), n, 1, d_m, 1, queue);
  // magma_setvector(batchCount, sizeof(magma_int_t), m, 1, d_n, 1, queue);
  // magma_setvector(batchCount, sizeof(magma_int_t), k, 1, d_k, 1, queue);
  // magma_setvector(batchCount, sizeof(magma_int_t), n, 1, d_lddb, 1, queue);
  // magma_setvector(batchCount, sizeof(magma_int_t), k, 1, d_ldda, 1, queue);
  // magma_setvector(batchCount, sizeof(magma_int_t), n, 1, d_lddc, 1, queue);
  
  // std::cout<<"maga set_vector of d vars finsihed..."<<"\n";


  // TESTING_CHECK( magma_malloc((void**)&d_m, (batchCount+1)*sizeof(magma_int_t)) );
  // TESTING_CHECK( magma_malloc((void**)&d_n, (batchCount+1)*sizeof(magma_int_t)) );
  // TESTING_CHECK( magma_malloc((void**)&d_k, (batchCount+1)*sizeof(magma_int_t)) );

  magmablas_dgemm_vbatched(transA,transB, n,
      /* magma_int_t * */         m,
      /* magma_int_t * */         k,
      /* double */                alpha,
      /* double const *const * */ dB_array,
      /* magma_int_t */           n,
      /* double const *const * */ dA_array,
      /* magma_int_t * */         k,
      /* double */                beta,
      /* double ** */             dC_array,
      /* magma_int_t * */         n,
      /* magma_int_t */           batchCount,
      /* magma_queue_t */         queue);

  return 2;
}

