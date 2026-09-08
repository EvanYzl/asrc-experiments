#include <torch/extension.h>

torch::Tensor forward_cuda(
    const torch::Tensor &edge_index_, const torch::Tensor &edge_type_,
    const torch::Tensor &relation_, const torch::Tensor &input_);

std::tuple<torch::Tensor, torch::Tensor> backward_cuda(
    const torch::Tensor &edge_index_, const torch::Tensor &edge_type_,
    const torch::Tensor &relation_, const torch::Tensor &input_,
    const torch::Tensor &output_, const torch::Tensor &output_grad_);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mul_sum_forward", &forward_cuda, "forward");
  m.def("mul_sum_backward", &backward_cuda, "backward");
}