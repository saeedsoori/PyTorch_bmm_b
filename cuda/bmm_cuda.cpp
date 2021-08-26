#include <torch/extension.h>

#include <vector>
#include <pybind11/pybind11.h>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>
#include "magma_v2.h"
#include "magma_lapack.h"

// CUDA forward declarations
namespace py = pybind11;


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

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



class BatchMatmul{

public:

	float ** A_array;
  	float ** B_array;
  	float ** C_array;

  	float const* * dA_array;
  	float const* * dB_array;
  	float **dC_array;

  	// int* m_magma;
  	// int* n_magma;
  	// int* k_magma;

	void set_pointers(
		torch::Tensor A,
    	torch::Tensor B,
    	torch::Tensor C,
    	int batchCount,
    	std::vector<int> offset_A,
    	std::vector<int> offset_B,
    	std::vector<int> offset_C){

  	magma_queue_t queue;
  	magma_device_t device;
  	// std::cout<<"initialization finsihed..."<<"\n";

  	magma_getdevice( &device );
  	magma_queue_create( device, &queue );

  	TESTING_CHECK( magma_malloc_cpu( (void**)&A_array, sizeof(float*)*batchCount ) );
  	TESTING_CHECK( magma_malloc_cpu( (void**)&B_array, sizeof(float*)*batchCount ) );
  	TESTING_CHECK( magma_malloc_cpu( (void**)&C_array, sizeof(float*)*batchCount ) );


  	for (int i = 0; i < batchCount; ++i)
  	{
    // std::cout<<"processing input tensor:"<< i<< " \n";

    	A_array[i] = (float *) A.data_ptr() + offset_A[i];
    	B_array[i] = (float *) B.data_ptr() + offset_B[i];
    	C_array[i] = (float *) C.data_ptr() + offset_C[i];
  	}




    TESTING_CHECK( magma_malloc( (void**)&dA_array, sizeof(float*)*batchCount ) );
  	TESTING_CHECK( magma_malloc( (void**)&dB_array, sizeof(float*)*batchCount ) );
  	TESTING_CHECK( magma_malloc( (void**)&dC_array, sizeof(float*)*batchCount ) );

  	magma_setvector(batchCount, sizeof(float*), A_array, 1, dA_array, 1, queue);
  	magma_setvector(batchCount, sizeof(float*), B_array, 1, dB_array, 1, queue);
  	magma_setvector(batchCount, sizeof(float*), C_array, 1, dC_array, 1, queue);

  	

  };




  int CublasForward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    std::vector<int> m, std::vector<int> n, std::vector<int> k, int batchCount,
    std::vector<int> offset_A,
    std::vector<int> offset_B,
    std::vector<int> offset_C
    ) {
    
    // int* m_arr = (int*) m.data_ptr();
    // int* n_arr = (int*) n.data_ptr();
    // int* k_arr = (int*) k.data_ptr();
	

  cublasStatus_t status;
  cudaStream_t *streams = (cudaStream_t *) malloc(batchCount*sizeof(cudaStream_t));

  if (streams == 0)
  {
      fprintf(stderr, "!!!! stream error\n");
    
  }

  for(int i=0; i<batchCount; i++)
    cudaStreamCreate(&streams[i]);

  cublasHandle_t handle;
  status = cublasCreate(&handle);



  float  alpha = 1.0f;
  float  beta = 0.0f;


  // Launch each DGEMM operation in own CUDA stream
#pragma omp parallel
#pragma omp for
for(int i=0; i<batchCount; i++){
    // Set CUDA stream
    cublasSetStream(handle, streams[i]);
   status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n[i], m[i], k[i], &alpha, (float *) (B.data_ptr() )+ offset_B[i],
                       n[i], (float *) (A.data_ptr() )+ offset_A[i], k[i], &beta, (float *) (C.data_ptr() )+ offset_C[i], n[i]);

}


    
};



int MagmaForward(
    torch::Tensor m_,
    torch::Tensor n_,
    torch::Tensor k_,
    int batch_size) {
  


  magma_int_t* d_m;
  magma_int_t* d_n;
  magma_int_t* d_k;

  float  alpha = 1.0;
  float  beta = 0.0;
  // magma_int_t* d_lddb;
  // magma_int_t* d_ldda;
  // magma_int_t* d_lddc;



  magma_trans_t transA = MagmaNoTrans;
  magma_trans_t transB = MagmaNoTrans;

  magma_int_t batchCount = batch_size;
  magma_queue_t queue;
  magma_device_t device;
  // std::cout<<"initialization finsihed..."<<"\n";

  magma_getdevice( &device );
  magma_queue_create( device, &queue );

  int* m = (int*) m_.data_ptr();
  int* n = (int*) n_.data_ptr();
  int* k = (int*) k_.data_ptr();

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


};





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	std::string name = std::string("BatchMatmul");

	py::class_<BatchMatmul>(m, name.c_str())
      .def(py::init<>())
      // .def("setKey", &Foo::setKey)
      .def("set_pointers", &BatchMatmul::set_pointers)
      .def("CublasForward", &BatchMatmul::CublasForward)
      .def("MagmaForward", &BatchMatmul::MagmaForward);
      // .def("getKey", &Foo::getKey);
  // m.def("forward", &bmm_forward, "BMM forward (CUDA)");
  // m.def("cublas_gemm_call", &cublas_forward, "BMM cublas_gemm_call (CUDA)");
}
