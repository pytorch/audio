#pragma once

#include <cuda_runtime_api.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/device.h>

namespace libtorchaudio::cuda {

inline cudaStream_t getCurrentCUDAStream(
    torch::stable::DeviceIndex device_index = -1) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

inline void setCurrentCUDAStream(
    cudaStream_t stream,
    torch::stable::DeviceIndex device_index = -1) {
  TORCH_ERROR_CODE_CHECK(
      torch_set_current_cuda_stream(static_cast<void*>(stream), device_index));
}

inline cudaStream_t getStreamFromPool(
    const bool isHighPriority = false,
    torch::stable::DeviceIndex device_index = -1) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(torch_get_cuda_stream_from_pool(
      isHighPriority, device_index, &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

inline void synchronize(
    cudaStream_t stream,
    torch::stable::DeviceIndex device_index = -1) {
  TORCH_ERROR_CODE_CHECK(
      torch_cuda_stream_synchronize(static_cast<void*>(stream), device_index));
}

} // namespace libtorchaudio::cuda
