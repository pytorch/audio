#include <iostream>
#include <numeric>

#include <torch/extension.h>
#include "rnnt.h"

int cpu_rnnt(torch::Tensor acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor grads,
            int blank_label,
            int num_threads);

int64_t cpu_rnnt_torchbind(torch::Tensor acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor grads,
            int64_t blank_label,
            int64_t num_threads) {
return cpu_rnnt(acts,
            labels,
            input_lengths,
            label_lengths,
            costs,
            grads,
            blank_label,
            num_threads);
}

TORCH_LIBRARY(warprnnt_pytorch_warp_rnnt, m) {
    m.def("rnnt(Tensor acts,"
               "Tensor labels,"
               "Tensor input_lengths,"
               "Tensor label_lengths,"
               "Tensor costs,"
               "Tensor grads,"
               "int blank_label,"
               "int num_threads) -> int");
}

TORCH_LIBRARY_IMPL(warprnnt_pytorch_warp_rnnt, CPU, m) {
    m.impl("rnnt", &cpu_rnnt_torchbind);
}
