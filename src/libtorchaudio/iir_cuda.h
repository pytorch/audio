#pragma once

#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor cuda_lfilter_core_loop(Tensor in, Tensor a_flipped, Tensor padded_out);
