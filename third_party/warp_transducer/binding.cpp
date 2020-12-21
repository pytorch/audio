#include <iostream>
#include <numeric>

#include <torch/extension.h>
#include "rnnt.h"

#ifdef WARPRNNT_ENABLE_GPU
    #include "THC.h"
    extern THCState* state;
#endif

int64_t cpu_rnnt(torch::Tensor acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor grads,
            int64_t blank_label,
            int64_t num_threads) {

    int maxT = acts.size(0);
    int maxU = acts.size(1);
    int minibatch_size = acts.size(2);
    int alphabet_size = acts.size(3);

	if (true) {
		minibatch_size = acts.size(0);
		maxT = acts.size(1);
		maxU = acts.size(2);
	}

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.batch_first = true;
    options.loc = RNNT_CPU;
    options.num_threads = num_threads;
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

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
#ifdef WARPRNNT_ENABLE_GPU
int64_t gpu_rnnt(torch::Tensor acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor grads,
            int64_t blank_label,
            int64_t num_threads) {

    int minibatch_size = acts.size(0);
    int maxT = acts.size(1);
    int maxU = acts.size(2);
    int alphabet_size = acts.size(3);

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.loc = RNNT_GPU;
    options.stream = at::cuda::getCurrentCUDAStream();
    options.num_threads = num_threads;
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    switch (acts.type().scalarType()) {
      case torch::ScalarType::Float:
        {
        size_t gpu_size_bytes;
        get_workspace_size(maxT, maxU, minibatch_size,
                           true, &gpu_size_bytes);

        cudaSetDevice(acts.get_device());

        void* gpu_workspace = THCudaMalloc(state, gpu_size_bytes);

        compute_rnnt_loss(acts.data<float>(), grads.data<float>(),
                         labels.data<int>(), label_lengths.data<int>(),
                         input_lengths.data<int>(), alphabet_size,
                         minibatch_size, costs.data<float>(),
                         gpu_workspace, options);

        THCudaFree(state, gpu_workspace);
        return 0;
        }
      case torch::ScalarType::Double:
        {
        size_t gpu_size_bytes;
        get_workspace_size(maxT, maxU, minibatch_size,
                           true, &gpu_size_bytes);

        cudaSetDevice(acts.get_device());

        void* gpu_workspace = THCudaMalloc(state, gpu_size_bytes);

        compute_rnnt_loss_fp64(acts.data<double>(), grads.data<double>(),
                         labels.data<int>(), label_lengths.data<int>(),
                         input_lengths.data<int>(), alphabet_size,
                         minibatch_size, costs.data<double>(),
                         gpu_workspace, options);

        THCudaFree(state, gpu_workspace);
        return 0;
        }
      default:
        std::cerr << __FILE__ << ':' << __LINE__ << ": " << "unsupported data type" << std::endl;
    }
    return -1;
}
#endif

TORCH_LIBRARY(warprnnt_pytorch_warp_rnnt, m) {
    m.def("cpu_rnnt", &cpu_rnnt);
#ifdef WARPRNNT_ENABLE_GPU
    m.def("gpu_rnnt", &gpu_rnnt);
#endif
}
