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

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#endif

using torch::stable::Tensor;

namespace torchaudio::stable {

using Layout = int32_t;

// TODO: When sizes and strides are implemented in torch::stable,
// eliminate sizes and strides function below.
inline std::vector<int64_t> sizes(const Tensor& t) {
  int64_t* ptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(t.get(), &ptr));
  std::vector<int64_t> r(ptr, ptr + t.dim());
  return r;
}

inline std::vector<int64_t> strides(const Tensor& t) {
  int64_t* ptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(t.get(), &ptr));
  std::vector<int64_t> r(ptr, ptr + t.dim());
  return r;
}

// TODO: When https://github.com/pytorch/pytorch/pull/161891 lands,
// eliminate mutable_data_ptr and const_data_ptr templates.
#define aoti_torch_get_mutable_data_ptr aoti_torch_get_data_ptr
#define aoti_torch_get_const_data_ptr aoti_torch_get_data_ptr
template <typename T>
T* mutable_data_ptr(const Tensor& t) {
  void* data_ptr{};
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_mutable_data_ptr(t.get(), &data_ptr));
  return reinterpret_cast<T*>(data_ptr);
}

template <typename T>
const T* const_data_ptr(const Tensor& t) {
  const void* data_ptr{};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_const_data_ptr(t.get(), const_cast<void**>(&data_ptr)));
  return reinterpret_cast<const T*>(data_ptr);
}

// TODO: When cpu is implemented in torch::stable, eliminate
// cpu function below.
inline Tensor cpu(const Tensor& self) {
  auto sizes_ = sizes(self);
  auto cpu_type = aoti_torch_device_type_cpu();
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

// TODO:
inline Tensor cuda(const Tensor& self, int32_t cuda_index) {
  auto sizes_ = sizes(self);
  auto cuda_type = aoti_torch_device_type_cuda();
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

// TODO: remove when torch::stable provides new_zeros
inline Tensor new_zeros(
    const Tensor& self,
    std::vector<int64_t> size,
    std::optional<c10::ScalarType> dtype = std::nullopt,
    std::optional<Layout> layout = std::nullopt,
    std::optional<torch::stable::Device> device = std::nullopt,
    std::optional<bool> pin_memory = std::nullopt) {
  int32_t target_dtype{};
  if (dtype.has_value()) {
    target_dtype = torch::stable::detail::to<int32_t>(
        torch::stable::detail::from(dtype.value()));
  } else {
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(self.get(), &target_dtype));
  }

  Layout layout_;
  if (layout.has_value()) {
    layout_ = layout.value();
  } else {
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(self.get(), &layout_));
  }

  int32_t device_type;
  torch::stable::DeviceIndex device_index = 0;
  if (device.has_value()) {
    auto device_ = device.value();
    device_type = static_cast<int32_t>(device_.type());
    device_index = device_.index();
  } else {
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_type(self.get(), &device_type));
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_index(self.get(), &device_index));
  }

  // TODO: pin_memory

  AtenTensorHandle ret0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_new_empty(
      self.get(),
      size.data(),
      static_cast<int64_t>(size.size()),
      &target_dtype,
      &layout_,
      &device_type,
      device_index,
      nullptr, // pin_memory (nullptr for default)
      &ret0));

  auto result = Tensor(ret0);
  torch::stable::zero_(result);
  return result;
}

// An analog of item template function defined in
// ATen/templates/TensorBody.h
template <typename T>
T item(const Tensor& self) {
  STD_TORCH_CHECK(
      self.numel() == 1, "item requires single element tensor input");
  if (self.is_cpu()) {
    return torchaudio::stable::const_data_ptr<T>(self)[0];
#ifdef USE_CUDA
  } else if (self.is_cuda()) {
    T value;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &value, self.data_ptr(), sizeof(T), cudaMemcpyDeviceToHost));
    return value;
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
