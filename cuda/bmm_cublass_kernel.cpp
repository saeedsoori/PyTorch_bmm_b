#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Includes, cuda */
#include <cublas_v2.h>
// #include <helper_cuda.h>

#include <string.h> //for memcpy

// #include <magma_v2.h>






int bmm_cublass_forward(
    float const* * dA_array,
    float const* * dB_array,
    float **dC_array,
    int* m,
    int* n,
    int* k,
    int batch_count,
    std::vector<int> offset_A,
    std::vector<int> offset_B,
    std::vector<int> offset_C) {
  


  cudaStream_t *streams = (cudaStream_t *) malloc(batch_count*sizeof(cudaStream_t));

  for(int i=0; i<batch_count; i++)
    cudaStreamCreate(&streams[i]);

  cublasHandle_t handle;
  status = cublasCreate(&handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }



  float  alpha = 1.0;
  float  beta = 0.0;

  // Launch each DGEMM operation in own CUDA stream
for(int i=0; i<batch_count; i++){
    // Set CUDA stream
    cublasSetStream(handle, streams[i]);

    // DGEMM: C = alpha*A*B + beta*C
    /* Performs operation using cublas */
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n[i], m[i], k[i], &alpha, dB_array[i],
                       n[i], dA_array[i], k[i], &beta, dC_array[i], n[i]);
}



  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }
  return 2;
}

