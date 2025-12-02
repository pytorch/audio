#pragma once
// TODO: remove this file once https://github.com/pytorch/pytorch/pull/169376
// has landed.

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/c/shim.h>

inline AOTITorchError tmp_torch_set_current_cuda_stream(
    void* stream,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(stream), device_index));
  });
}

inline AOTITorchError tmp_torch_get_cuda_stream_from_pool(
    const bool isHighPriority,
    int32_t device_index,
    void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(cudaStream_t*)(ret_stream) =
        at::cuda::getStreamFromPool(isHighPriority, device_index);
  });
}

inline AOTITorchError tmp_torch_cuda_stream_synchronize(
    void* stream,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(stream), device_index)
        .synchronize();
  });
}
