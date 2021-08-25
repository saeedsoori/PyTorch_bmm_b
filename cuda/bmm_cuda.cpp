#include <torch/extension.h>

#include <vector>
#include <pybind11/pybind11.h>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
// CUDA forward declarations
namespace py = pybind11;




int bmm_cublass_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int* m,
    int* n,
    int* k,
    int batch_size,
    std::vector<int> offset_A,
    std::vector<int> offset_B,
    std::vector<int> offset_C);



// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// int bmm_forward(
//     std::vector<torch::Tensor> A,
//     std::vector<torch::Tensor> B,
//     std::vector<torch::Tensor> C,
//     torch::Tensor m, torch::Tensor n, torch::Tensor k, int batch_size) {
    
//     int* m_arr = (int*) m.data_ptr();
//     int* n_arr = (int*) n.data_ptr();
//     int* k_arr = (int*) k.data_ptr();


    
//   return bmm_cuda_forward(A, B, C, m_arr ,n_arr , k_arr, batch_size);
// }

class Foo {

public:
  float ** A_array;
  float ** B_array;
  float ** C_array;

  float const* * dA_array;
  float const* * dB_array;
  float **dC_array;


  int x;

  // void setKey(torch::Tensor A, torch::Tensor B, torch::Tensor C,
  //  std::vector<int> offset_A,
  //   std::vector<int> offset_B,
  //   std::vector<int> offset_C);

  void setKey(int i){
  	x = i;
  };

  int getKey(){
  	return x;
  };

  
  void set_pointers_cublas(torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int batchCount,
    std::vector<int> offset_A,
    std::vector<int> offset_B,
    std::vector<int> offset_C){

  cublasStatus_t status;

  	A_array = (float **) malloc(batchCount*sizeof(float*));
  	B_array = (float **) malloc(batchCount*sizeof(float*));
  	C_array = (float **) malloc(batchCount*sizeof(float*));

  	if (A_array==0 || B_array ==0 || C_array ==0)
  	{
  		fprintf(stderr, "!!!! host memory allocation error \n");
  	}

    	std::cout<<" host memory initialization...\n";

  	for (int i = 0; i < batchCount; ++i)
  	{
    // std::cout<<"processing input tensor:"<< i<< " \n";

    	A_array[i] = (float *) A.data_ptr() + offset_A[i];
    	B_array[i] = (float *) B.data_ptr() + offset_B[i];
    	C_array[i] = (float *) C.data_ptr() + offset_C[i];

    	std::cout<< A_array[i] << " " << B_array[i] <<" "<<C_array[i]<<"\n";
  	}
    	std::cout<<" host memory initialized...\n";


  	if (cudaMalloc(reinterpret_cast<void **>(&dA_array), batchCount * sizeof(float*)) !=
      cudaSuccess) {
    	fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
  	}
  	if (cudaMalloc(reinterpret_cast<void **>(&dB_array), batchCount * sizeof(float*)) !=
      cudaSuccess) {
    	fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
  	}
  	if (cudaMalloc(reinterpret_cast<void **>(&dC_array), batchCount * sizeof(float*)) !=
      cudaSuccess) {
    	fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
  	}

  	/* Initialize the device matrices with the host matrices */
  status = cublasSetVector(batchCount, sizeof(float*), A_array, 1, dA_array, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write A)\n");
  }

  status = cublasSetVector(batchCount, sizeof(float*), B_array, 1, dB_array, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write B)\n");
  }

  status = cublasSetVector(batchCount, sizeof(float*), C_array, 1, dC_array, 1);




  std::cout<<" device memory initialized...\n";


  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write C)\n");
  }


  	

  };

  

  int fooCublasforward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor m, torch::Tensor n, torch::Tensor k, int batchCount,
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
  // std::cout<<"H2\n";

  for(int i=0; i<batchCount; i++)
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



// status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n[0], m[0], k[0], &alpha, dB_array[0],
//                        n[0], dA_array[0], k[0], &beta, dC_array[0], n[0]);

  // Launch each DGEMM operation in own CUDA stream
for(int i=0; i<batchCount; i++){
    // Set CUDA stream
    cublasSetStream(handle, streams[i]);
    // std::cout<<"H7\n";

    // DGEMM: C = alpha*A*B + beta*C
    /* Performs operation using cublas */
  // status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n[i], m[i], k[i], &alpha, reinterpret_cast<float *> (B.data_ptr() )+ offset_B[i],
  //                      n[i], reinterpret_cast<float *> (A.data_ptr() )+ offset_A[i], k[i], &beta, reinterpret_cast<float *> (C.data_ptr() )+ offset_C[i], n[i]);
  

   status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n[i], m[i], k[i], &alpha, reinterpret_cast<float *> (B.data_ptr() )+ offset_B[i],
                       n[i], reinterpret_cast<float *> (A.data_ptr() )+ offset_A[i], k[i], &beta, reinterpret_cast<float *> (C.data_ptr() )+ offset_C[i], n[i]);

  // status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, *(reinterpret_cast<int*> (n.data_ptr())+i), *(reinterpret_cast<int*> (m.data_ptr())+i), *(reinterpret_cast<int*> (k.data_ptr())+i), &alpha, reinterpret_cast<float *> (B.data_ptr() )+ offset_B[i],
  //                      *(reinterpret_cast<int*> (n.data_ptr())+i), reinterpret_cast<float *> (A.data_ptr() )+ offset_A[i], *(reinterpret_cast<int*> (k.data_ptr())+i), &beta, reinterpret_cast<float *> (C.data_ptr() )+ offset_C[i], *(reinterpret_cast<int*> (n.data_ptr())+i));
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
    
};


};





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	std::string name = std::string("Foo");

	py::class_<Foo>(m, name.c_str())
      .def(py::init<>())
      .def("setKey", &Foo::setKey)
      .def("set_pointers_cublas", &Foo::set_pointers_cublas)
      .def("fooCublasforward", &Foo::fooCublasforward)
      .def("getKey", &Foo::getKey);
  // m.def("forward", &bmm_forward, "BMM forward (CUDA)");
  // m.def("cublas_gemm_call", &cublas_forward, "BMM cublas_gemm_call (CUDA)");
}
