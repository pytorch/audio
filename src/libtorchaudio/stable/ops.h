#pragma once

/*
  This header files provides torchaudio::stable operations that are
  torch::stable::Tensor-compatible analogus operations defined in
  ATen/core/TensorBase.h and elsewhere.

  TODO: remove this header file when torch::stable provides all
  features implemented here.
*/

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

namespace torchaudio::stable {

using torch::stable::Tensor;

// TODO: When cpu op is implemented in torch::stable, eliminate cpu
// function below.
inline Tensor cpu(const Tensor& self) {
  auto sizes_ = self.sizes();
  int32_t cpu_type = static_cast<int32_t>(torch::stable::DeviceType::CPU);
  int32_t dtype;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(self.get(), &dtype));
  int32_t layout;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(self.get(), &layout));
  AtenTensorHandle ret0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_new_empty(
      self.get(),
      sizes_.data(),
      static_cast<int64_t>(self.dim()),
      &dtype,
      &layout,
      &cpu_type,
      0,
      nullptr, // pin_memory (nullptr for default)
      &ret0));
  auto result = Tensor(ret0);
  copy_(result, self);
  return result;
}

// TODO: When cuda op is implemented in torch::stable, eliminate cuda
// function below.
inline Tensor cuda(const Tensor& self, int32_t cuda_index) {
  auto sizes_ = self.sizes();
  int32_t cuda_type = static_cast<int32_t>(torch::stable::DeviceType::CUDA);
  int32_t dtype;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(self.get(), &dtype));
  int32_t layout;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(self.get(), &layout));
  AtenTensorHandle ret0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_new_empty(
      self.get(),
      sizes_.data(),
      static_cast<int64_t>(self.dim()),
      &dtype,
      &layout,
      &cuda_type,
      cuda_index,
      nullptr, // pin_memory (nullptr for default)
      &ret0));
  auto result = Tensor(ret0);
  copy_(result, self);
  return result;
}

// An analog of item template function defined in
// ATen/templates/TensorBody.h
template <typename T>
T item(const Tensor& self) {
  STD_TORCH_CHECK(
      self.numel() == 1, "item requires single element tensor input");
  if (self.is_cpu()) {
    // TODO: use `return self.const_data_ptr<T>()[0];` after torch
    // stable supports const_data_ptr templates.
    return reinterpret_cast<const T*>(self.const_data_ptr())[0];
#ifdef USE_CUDA
  } else if (self.is_cuda()) {
    return torchaudio::stable::item<T>(torchaudio::stable::cpu(self));
#endif
  } else {
    STD_TORCH_CHECK(false, "unreachable"); // not implemented
  }
}

inline Tensor unsqueeze(const Tensor& self, int64_t dim) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dim)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::unsqueeze", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(index)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::select", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline Tensor squeeze(const Tensor& self, int64_t dim) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dim)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::squeeze", "dim", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline Tensor matmul(const Tensor& self, const Tensor& other) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(other)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::matmul", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline Tensor subtract(const Tensor& self, const Tensor& other) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(other)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::subtract", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

} // namespace torchaudio::stable
