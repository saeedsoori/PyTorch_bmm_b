#include <torch/extension.h>

#include <vector>
#include <pybind11/pybind11.h>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
// CUDA forward declarations
namespace py = pybind11;




// int bmm_cublass_forward(
//     torch::Tensor A,
//     torch::Tensor B,
//     torch::Tensor C,
//     std::vector<int> m,
//     std::vector<int> n,
//     std::vector<int> k,
//     int batch_size,
//     std::vector<int> offset_A,
//     std::vector<int> offset_B,
//     std::vector<int> offset_C);



// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



class cublas_class {
public:
  int Cublasforward(
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
for(int i=0; i<batchCount; i++){
    // Set CUDA stream
    cublasSetStream(handle, streams[i]);
   status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n[i], m[i], k[i], &alpha, reinterpret_cast<float *> (B.data_ptr() )+ offset_B[i],
                       n[i], reinterpret_cast<float *> (A.data_ptr() )+ offset_A[i], k[i], &beta, reinterpret_cast<float *> (C.data_ptr() )+ offset_C[i], n[i]);

}



  // if (status != CUBLAS_STATUS_SUCCESS) {
  //   fprintf(stderr, "!!!! kernel execution error.\n");
  //   return EXIT_FAILURE;
  // }
    
};


};





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	std::string name = std::string("cublas_class");

	py::class_<cublas_class>(m, name.c_str())
      .def(py::init<>())
      // .def("setKey", &Foo::setKey)
      // .def("set_pointers_cublas", &Foo::set_pointers_cublas)
      .def("Cublasforward", &cublas_class::Cublasforward);
      // .def("getKey", &Foo::getKey);
  // m.def("forward", &bmm_forward, "BMM forward (CUDA)");
  // m.def("cublas_gemm_call", &cublas_forward, "BMM cublas_gemm_call (CUDA)");
}
