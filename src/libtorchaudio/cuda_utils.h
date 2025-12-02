#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/device.h>

#include <cuda_runtime_api.h>

#define TA_CUDA_CHECK(...) __VA_ARGS__

namespace libtorchaudio::cuda {

inline cudaStream_t getCurrentCUDAStream(
    torch::stable::DeviceIndex device_index = -1) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

// A strip-down version of at::cuda::stream_synchronize
inline void stream_synchronize(cudaStream_t stream) {
  TA_CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace libtorchaudio::cuda
