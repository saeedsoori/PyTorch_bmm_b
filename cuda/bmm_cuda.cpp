#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

int bmm_cuda_forward(
    double* pA,
    double* pB,
    int m,
    int n,
    int k);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int bmm_forward(
    torch::Tensor A,
    torch::Tensor B, int m, int n, int k) {
    double *pA = (double *) A.data_ptr();
    double *pB = (double *) B.data_ptr();
    std::cout<<"here\n";
    std::cout<<m<<" "<<n<<" "<<k<<"\n";
  // CHECK_INPUT(A);
  // CHECK_INPUT(B);


  return bmm_cuda_forward(pA, pB, m ,n, k);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bmm_forward, "BMM forward (CUDA)");
}
