#pragma once
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/ops.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#endif

namespace torchaudio {

namespace util {
  inline std::vector<int64_t> sizes(const torch::stable::Tensor& t) {
    auto sizes_ = t.sizes();
    std::vector<int64_t> sizes(sizes_.data(), sizes_.data() + t.dim());
    return sizes;
  }

  template <typename T>
  T item(const torch::stable::Tensor& t) {
    STD_TORCH_CHECK(t.numel() == 1, "item requires single element tensor input");
    if (t.is_cpu()) {
      return t.const_data_ptr<T>()[0];
#ifdef USE_CUDA
    } else if (t.is_cuda()) {
      T value;
      C10_CUDA_CHECK(cudaMemcpyAsync(&value, t.data_ptr(), sizeof(T), cudaMemcpyDeviceToHost));
      return value;
#endif
    } else {
      STD_TORCH_CHECK(false, "unreachable");
    }
  }

  template <typename T>
  T max(const torch::stable::Tensor& t) {
    // TODO: eliminate const_cast after pytorch/pytorch#161826 is fixed
    return item<T>(torch::stable::amax(const_cast<torch::stable::Tensor&>(t), {}));
  }
}

bool is_align_available();
std::optional<int64_t> cuda_version();
} // namespace torchaudio
