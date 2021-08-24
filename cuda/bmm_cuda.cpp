#include <torch/extension.h>

#include <vector>
#include <pybind11/pybind11.h>
// includes, project
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




int bmm_cuda_forward(
    float const* * dA_array,
    float const* * dB_array,
    float ** dC_array,
    int* m,
    int* n,
    int* k,
    int batch_size,
    std::vector<int> offset_A,
    std::vector<int> offset_B,
    std::vector<int> offset_C);


// int bmm_cuda_forward(
    // std::vector<torch::Tensor> A,
    // std::vector<torch::Tensor> B,
    // std::vector<torch::Tensor> C,
    // int* m,
    // int* n,
    // int* k,
    // int batch_size);


int bmm_cublass_forward(
    float const* * dA_array,
    float const* * dB_array,
    float ** dC_array,
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

  void set_pointers(torch::Tensor A,
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

  int fooforward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor m, torch::Tensor n, torch::Tensor k, int batchCount,
    std::vector<int> offset_A,
    std::vector<int> offset_B,
    std::vector<int> offset_C
    ) {
    
    int* m_arr = (int*) m.data_ptr();
    int* n_arr = (int*) n.data_ptr();
    int* k_arr = (int*) k.data_ptr();
	

    
    
  return bmm_cuda_forward(dA_array, dB_array, dC_array, m_arr ,n_arr , k_arr, batchCount, offset_A, offset_B, offset_C);
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
    
    int* m_arr = (int*) m.data_ptr();
    int* n_arr = (int*) n.data_ptr();
    int* k_arr = (int*) k.data_ptr();
	

    
    
  return bmm_cublass_forward(dA_array, dB_array, dC_array, m_arr ,n_arr , k_arr, batchCount, offset_A, offset_B, offset_C);
};


};


// int bmm_forward(
//     torch::Tensor A,
//     torch::Tensor B,
//     torch::Tensor C,
//     torch::Tensor m, torch::Tensor n, torch::Tensor k, int batch_size,
//     std::vector<int> offset_A,
//     std::vector<int> offset_B,
//     std::vector<int> offset_C
//     ) {
    
//     int* m_arr = (int*) m.data_ptr();
//     int* n_arr = (int*) n.data_ptr();
//     int* k_arr = (int*) k.data_ptr();

    
    
//   return bmm_cuda_forward(A, B, C, m_arr ,n_arr , k_arr, batch_size, offset_A, offset_B, offset_C);
// }

// int cublas_forward(

//     torch::Tensor A,
//     torch::Tensor B,
//     torch::Tensor C,
//     int* m,
//     int* n,
//     int* k,
//     int batch_size,
//     std::vector<int> offset_A,
//     std::vector<int> offset_B,
//     std::vector<int> offset_C) {
// 	// CHECK_CONTIGUOUS(A);
// 	// CHECK_CONTIGUOUS(B);
// 	// CHECK_CONTIGUOUS(C);

//     // double *pA = (double *) A[0].data_ptr();
//     // double *pB = (double *) B[0].data_ptr();

//     // std::cout<<"elements in A: "<<A.size()<<"\n";
//     // std::cout<<"elements in B: "<<B.size()<<"\n";

//     // int* m_arr = (int*) malloc (2*sizeof(int));
//     // int* n_arr = (int*) malloc (2*sizeof(int));
//     // int* k_arr = (int*) malloc (2*sizeof(int));

    

//     // for (int i = 0; i < A.size(); ++i)
//     // {
    	
//     // }

//     // int* m_arr = (int*) m.data_ptr();
//     // int* n_arr = (int*) n.data_ptr();
//     // int* k_arr = (int*) k.data_ptr();
//     // std::cout<<"here\n";
//     // std::cout<<m_arr[0]<<" "<<n_arr[0]<<" "<<k_arr[0]<<"\n";

//   // CHECK_INPUT(A);
//   // CHECK_INPUT(B);


//   return cublas_gemm_call(A, B, C, m ,n , k, batch_size, offset_A, offset_B, offset_C);
// }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	std::string name = std::string("Foo");

	py::class_<Foo>(m, name.c_str())
      .def(py::init<>())
      .def("setKey", &Foo::setKey)
      .def("set_pointers", &Foo::set_pointers)
      .def("fooforward", &Foo::fooforward)
      .def("fooCublasforward", &Foo::fooCublasforward)
      .def("getKey", &Foo::getKey);
  // m.def("forward", &bmm_forward, "BMM forward (CUDA)");
  // m.def("cublas_gemm_call", &cublas_forward, "BMM cublas_gemm_call (CUDA)");
}
