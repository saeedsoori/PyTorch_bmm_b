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







int bmm_cuda_forward(
    float const* * dA_array,
    float const* * dB_array,
    float **dC_array,
    int* m,
    int* n,
    int* k,
    int batch_size,
    std::vector<int> offset_A,
    std::vector<int> offset_B,
    std::vector<int> offset_C) {
  

  // float ** hA_array;
  // float ** hB_array;
  // float ** hC_array;

  // float const* * dA_array;
  // float const* * dB_array;
  // float **dC_array;

  magma_int_t* d_m;
  magma_int_t* d_n;
  magma_int_t* d_k;

  float  alpha = 1.0;
  float  beta = 0.0;
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



  // TESTING_CHECK( magma_malloc_cpu( (void**)&hA_array, sizeof(float*)*batchCount ) );
  // TESTING_CHECK( magma_malloc_cpu( (void**)&hB_array, sizeof(float*)*batchCount ) );
  // TESTING_CHECK( magma_malloc_cpu( (void**)&hC_array, sizeof(float*)*batchCount ) );


  // for (int i = 0; i < batchCount; ++i)
  // {
  //   // std::cout<<"processing input tensor:"<< i<< " \n";

  //   hA_array[i] = (float *) A.data_ptr() + offset_A[i];
  //   hB_array[i] = (float *) B.data_ptr() + offset_B[i];
  //   hC_array[i] = (float *) C.data_ptr() + offset_C[i];
  // }



  // TESTING_CHECK( magma_malloc( (void**)&dA_array, sizeof(float*)*batchCount ) );
  // TESTING_CHECK( magma_malloc( (void**)&dB_array, sizeof(float*)*batchCount ) );
  // TESTING_CHECK( magma_malloc( (void**)&dC_array, sizeof(float*)*batchCount ) );

  // magma_setvector(batchCount, sizeof(float*), hA_array, 1, dA_array, 1, queue);
  // magma_setvector(batchCount, sizeof(float*), hB_array, 1, dB_array, 1, queue);
  // magma_setvector(batchCount, sizeof(float*), hC_array, 1, dC_array, 1, queue);

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

  magmablas_sgemm_vbatched(transA,transB, n,
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

