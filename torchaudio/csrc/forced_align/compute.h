#pragma once

#include <torch/script.h>

std::tuple<torch::Tensor, torch::Tensor> forced_align(
    const torch::Tensor& logProbs,
    const torch::Tensor& targets,
    const torch::Tensor& inputLengths,
    const torch::Tensor& targetLengths,
    const int64_t blank);
