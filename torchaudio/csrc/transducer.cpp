#include <iostream>
#include <numeric>

#include <torch/extension.h>
#include "rnnt.h"

int64_t cpu_rnnt_loss(torch::Tensor acts,
                      torch::Tensor labels,
                      torch::Tensor input_lengths,
                      torch::Tensor label_lengths,
                      torch::Tensor costs,
                      torch::Tensor grads,
                      int64_t blank_label,
                      int64_t num_threads) {

    int maxT = acts.size(1);
    int maxU = acts.size(2);
    int minibatch_size = acts.size(0);
    int alphabet_size = acts.size(3);

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.batch_first = true;
    options.loc = RNNT_CPU;
    options.num_threads = num_threads;

    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);

    size_t cpu_size_bytes = 0;
    switch (acts.type().scalarType()) {
      case torch::ScalarType::Float:
        {
        get_workspace_size(maxT, maxU, minibatch_size,
                           false, &cpu_size_bytes);

        float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];
        compute_rnnt_loss(acts.data<float>(), grads.data<float>(),
                         labels.data<int>(), label_lengths.data<int>(),
                         input_lengths.data<int>(), alphabet_size,
                         minibatch_size, costs.data<float>(),
                         cpu_workspace, options);

        delete cpu_workspace;
        return 0;
        }
      case torch::ScalarType::Double:
        {
        get_workspace_size(maxT, maxU, minibatch_size,
                           false, &cpu_size_bytes,
                           sizeof(double));

        double* cpu_workspace = (double*) new unsigned char[cpu_size_bytes];
        compute_rnnt_loss_fp64(acts.data<double>(), grads.data<double>(),
                         labels.data<int>(), label_lengths.data<int>(),
                         input_lengths.data<int>(), alphabet_size,
                         minibatch_size, costs.data<double>(),
                         cpu_workspace, options);

        delete cpu_workspace;
        return 0;
        }
      default:
        std::cerr << __FILE__ << ':' << __LINE__ << ": " << "unsupported data type" << std::endl;
    }
    return -1;
}

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
    m.impl("rnnt_loss", &cpu_rnnt_loss);
}
