#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> bmm_cuda_forward(
    torch::Tensor A,
    torch::Tensor B);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> bmm_forward(
    torch::Tensor A,
    torch::Tensor B) {
    auto pA = A.data_ptr;
    auto pB = B.data_ptr;
    std::cout<<"here\n";
  // CHECK_INPUT(A);
  // CHECK_INPUT(B);


  return bmm_cuda_forward(A, B);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bmm_forward, "BMM forward (CUDA)");
}
