#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

int bmm_cuda_forward(
    double* pA,
    double* pB,
    int* m,
    int* n,
    int* k);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int bmm_forward(
    torch::Tensor A,
    torch::Tensor B, List[int] m, List[int] n, List[int] k) {
    double *pA = (double *) A.data_ptr();
    double *pB = (double *) B.data_ptr();
    int* m_arr = (int* )malloc(size_of(int)*m.size(0))
    int* n_arr = (int* )malloc(size_of(int)*m.size(0))
    int* k_arr = (int* )malloc(size_of(int)*m.size(0))
    std::cout<<"here\n";
    std::cout<<m_arr[0]<<" "<<n_arr[0]<<" "<<k_arr[0]<<"\n";
  // CHECK_INPUT(A);
  // CHECK_INPUT(B);


  return bmm_cuda_forward(pA, pB, m_arr ,n_arr , k_arr);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bmm_forward, "BMM forward (CUDA)");
}
