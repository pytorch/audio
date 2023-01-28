#include <torch/extension.h>

void cuda_lfilter_core_loop(
    const torch::Tensor& in,
    const torch::Tensor& a_flipped,
    torch::Tensor& padded_out);