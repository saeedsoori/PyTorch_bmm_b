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
#include <cublas_v2.h>

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

// int bmm_cublass_forward(
//     torch::Tensor A,
//     torch::Tensor B,
//     torch::Tensor C,
//     int* m,
//     int* n,
//     int* k,
//     int batch_count,
//     std::vector<int> offset_A,
//     std::vector<int> offset_B,
//     std::vector<int> offset_C) {
//   cublasStatus_t status;

// float *h_A;
//   float *h_B;
//   float *h_C;
//   float *h_C_ref;
//   float *d_A = 0;
//   float *d_B = 0;
//   float *d_C = 0;
//   float alpha = 1.0f;
//   float beta = 0.0f;
//   int N = 32;
//   int n2 = N * N;
//   int i;
//   float error_norm;
//   float ref_norm;
//   float diff;
//   cublasHandle_t handle;



//   /* Initialize CUBLAS */
//   printf("simpleCUBLAS test running..\n");

//   status = cublasCreate(&handle);

//   if (status != CUBLAS_STATUS_SUCCESS) {
//     fprintf(stderr, "!!!! CUBLAS initialization error\n");
//     return EXIT_FAILURE;
//   }
//   std::cout<<"\n";
//   /* Allocate host memory for the matrices */
//   h_A = reinterpret_cast<float *>(malloc(32 * sizeof(float*)));

//   if (h_A == 0) {
//     fprintf(stderr, "!!!! host memory allocation error (A)\n");
//     return EXIT_FAILURE;
//   }

//   h_B = reinterpret_cast<float *>(malloc(32 * sizeof(float*)));

//   if (h_B == 0) {
//     fprintf(stderr, "!!!! host memory allocation error (B)\n");
//     return EXIT_FAILURE;
//   }

//   h_C = reinterpret_cast<float *>(malloc(32 * sizeof(float*)));

//   if (h_C == 0) {
//     fprintf(stderr, "!!!! host memory allocation error (C)\n");
//     return EXIT_FAILURE;
//   }

//   /* Fill the matrices with test data */
//   for (i = 0; i < n2; i++) {
//     h_A[i] = rand() / static_cast<float>(RAND_MAX);
//     h_B[i] = rand() / static_cast<float>(RAND_MAX);
//     h_C[i] = rand() / static_cast<float>(RAND_MAX);
//   }

  
  

// // status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 32, 32, 32, &alpha, dB_array[0],
// //                        32, dA_array[0], 32, &beta, dC_array[0], 32);

//   status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 32, 32, 32, &alpha, (float *) B.data_ptr() + offset_B[0],
//                        32, (float *) A.data_ptr() + offset_A[0], 32, &beta, (float *) C.data_ptr() + offset_C[0], 32);

// std::cout<<"hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh\n";


//   /* Performs operation using cublas */
//   // status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A,
//                        // N, d_B, N, &beta, d_C, N);

//   // if (status != CUBLAS_STATUS_SUCCESS) {
//   //   fprintf(stderr, "!!!! kernel execution error.\n");
//   //   return EXIT_FAILURE;
//   // }

//   // /* Allocate host memory for reading back the result from device memory */
//   // h_C = reinterpret_cast<float *>(malloc(n2 * sizeof(h_C[0])));

//   // if (h_C == 0) {
//   //   fprintf(stderr, "!!!! host memory allocation error (C)\n");
//   //   return EXIT_FAILURE;
//   // }

//   // /* Read the result back */
//   // status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

//   // if (status != CUBLAS_STATUS_SUCCESS) {
//   //   fprintf(stderr, "!!!! device access error (read C)\n");
//   //   return EXIT_FAILURE;
//   // }

//   /* Check result against reference */
//   // error_norm = 0;
//   // ref_norm = 0;

//   // for (i = 0; i < n2; ++i) {
//   //   diff = h_C_ref[i] - h_C[i];
//   //   error_norm += diff * diff;
//   //   ref_norm += h_C_ref[i] * h_C_ref[i];
//   // }

//   // error_norm = static_cast<float>(sqrt(static_cast<double>(error_norm)));
//   // ref_norm = static_cast<float>(sqrt(static_cast<double>(ref_norm)));

//   // if (fabs(ref_norm) < 1e-7) {
//   //   fprintf(stderr, "!!!! reference norm is 0\n");
//   //   return EXIT_FAILURE;
//   // }

//   /* Memory clean up */
//   // free(h_A);
//   // free(h_B);
//   // free(h_C);
//   // free(h_C_ref);

//   // if (cudaFree(d_A) != cudaSuccess) {
//   //   fprintf(stderr, "!!!! memory free error (A)\n");
//   //   return EXIT_FAILURE;
//   // }

//   // if (cudaFree(d_B) != cudaSuccess) {
//   //   fprintf(stderr, "!!!! memory free error (B)\n");
//   //   return EXIT_FAILURE;
//   // }

//   // if (cudaFree(d_C) != cudaSuccess) {
//   //   fprintf(stderr, "!!!! memory free error (C)\n");
//   //   return EXIT_FAILURE;
//   // }

//   // /* Shutdown */
//   // status = cublasDestroy(handle);

//   // if (status != CUBLAS_STATUS_SUCCESS) {
//   //   fprintf(stderr, "!!!! shutdown error (A)\n");
//   //   return EXIT_FAILURE;
//   // }

//   // if (error_norm / ref_norm < 1e-6f) {
//   //   printf("simpleCUBLAS test passed.\n");
//   //   exit(EXIT_SUCCESS);
//   // } else {
//   //   printf("simpleCUBLAS test failed.\n");
//   //   exit(EXIT_FAILURE);
//   // }

// return 0;
// }








int bmm_cublass_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int* m,
    int* n,
    int* k,
    int batch_count,
    std::vector<int> offset_A,
    std::vector<int> offset_B,
    std::vector<int> offset_C) {
  
  cublasStatus_t status;
  // std::cout<<"H1\n";

  // status = cublasCreate(&handle);

//   if (status != CUBLAS_STATUS_SUCCESS) {
//     fprintf(stderr, "!!!! CUBLAS initialization error\n");
//     return EXIT_FAILURE;
//   }

  cudaStream_t *streams = (cudaStream_t *) malloc(batch_count*sizeof(cudaStream_t));

  if (streams == 0)
  {
      fprintf(stderr, "!!!! stream error\n");
    
  }
  // std::cout<<"H2\n";

  for(int i=0; i<batch_count; i++)
    cudaStreamCreate(&streams[i]);

  // std::cout<<"H3\n";

  cublasHandle_t handle;
  status = cublasCreate(&handle);



  // std::cout<<"H4\n";

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  // std::cout<<"H5\n";



  float  alpha = 1.0f;
  float  beta = 0.0f;

//   for(int i=0; i<batch_count; i++){
//   float *x = (float *) malloc(batch_count*sizeof(float *));
    
//     std::cout<< sizeof(n[i])<<"\n";
//     std::cout<< sizeof(m[i])<<"\n";
//     std::cout<< sizeof(k[i])<<"\n";
//     std::cout<< sizeof(dB_array)<<"\n";
//     std::cout<< sizeof(dA_array)<<"\n";
//     std::cout<< sizeof(dC_array)<<"\n";
//     std::cout<< "x:"<<sizeof(x)<<"\n";

//     // DGEMM: C = alpha*A*B + beta*C
//     /* Performs operation using cublas */
//   // status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n[i], m[i], k[i], &alpha, dB_array[i],
//                        // n[i], dA_array[i], k[i], &beta, dC_array[i], n[i]);
//   std::cout<<"H6\n";

// }

// status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n[0], m[0], k[0], &alpha, dB_array[0],
//                        n[0], dA_array[0], k[0], &beta, dC_array[0], n[0]);

  // Launch each DGEMM operation in own CUDA stream
for(int i=0; i<batch_count; i++){
    // Set CUDA stream
    cublasSetStream(handle, streams[i]);
    // std::cout<<"H7\n";

    // DGEMM: C = alpha*A*B + beta*C
    /* Performs operation using cublas */
  // status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n[i], m[i], k[i], &alpha, reinterpret_cast<float *> (B.data_ptr() )+ offset_B[i],
  //                      n[i], reinterpret_cast<float *> (A.data_ptr() )+ offset_A[i], k[i], &beta, reinterpret_cast<float *> (C.data_ptr() )+ offset_C[i], n[i]);
  
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &alpha, reinterpret_cast<float *> (B.data_ptr() )+ offset_B[i],
                       8, reinterpret_cast<float *> (A.data_ptr() )+ offset_A[i], 8, &beta, reinterpret_cast<float *> (C.data_ptr() )+ offset_C[i], 8);
// status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 32, 32, 32, &alpha, (float *) B.data_ptr() + offset_B[i],
//                        32, (float *) A.data_ptr() + offset_A[i], 32, &beta, (float *) C.data_ptr() + offset_C[i], 32);
  
// status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &alpha, reinterpret_cast<float *> (B.data_ptr() )+ offset_B[i],
                       // 8, reinterpret_cast<float *>(A.data_ptr()) + offset_A[i], 8, &beta,  reinterpret_cast<float *> (C.data_ptr() )+ offset_C[i], 8);

  // std::cout<<"H8\n";
// 
}



  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }
  return 2;
}

