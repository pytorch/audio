#pragma once

#ifdef _WIN32_TMP_DISABLE
// Workaround to linker error on Windows platform:
// LINK : error LNK2001: unresolved external symbol PyInit_...
#define TORCHAUDIO_EXT_MODULE(name)                               \
  static struct PyModuleDef name##_module = {                     \
      /*.m_base =*/PyModuleDef_HEAD_INIT, /*.m_name =*/"" #name}; \
  PyMODINIT_FUNC PyInit_##name(void) {                            \
    return PyModuleDef_Init(&name##_module);                      \
  }
#ifndef Py_LIMITED_API
#error "This extension module expects Py_LIMITED_API defined."
#endif
#include <Python.h>
#else
#define TORCHAUDIO_EXT_MODULE(name)
#endif

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/TensorAccessor.h>

namespace torchaudio {

namespace util {
template <typename T>
T max(const torch::stable::Tensor& t) {
  torch::stable::Tensor cpu = torch::stable::to(
      torch::stable::amax(t, {}),
      torch::headeronly::CppTypeToScalarType<T>::value,
      std::nullopt,
      torch::stable::Device("cpu"));
  return (cpu.const_data_ptr<T>())[0];
}
} // namespace util

bool is_align_available();
std::optional<int64_t> cuda_version();

template <typename T, size_t N>
using TensorAccessor = torch::headeronly::HeaderOnlyTensorAccessor<T, N>;

// TODO: eliminate accessor<T, N>(t) in favor of t.accessor<T, N>
// after Tensor::accessor is supported in stable ABI
template <typename T, size_t N>
inline TensorAccessor<T, N> accessor(torch::stable::Tensor t) {
  return TensorAccessor<T, N>(
      reinterpret_cast<T*>(t.data_ptr()), t.sizes().data(), t.strides().data());
}

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T, size_t N>
using PackedTensorAccessor32 =
    torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<
        T,
        N,
        torch::headeronly::RestrictPtrTraits,
        int32_t>;

// TODO: eliminate accessor<T, N>(t) in favor of t.accessor<T, N>
// after Tensor::accessor is supported in stable ABI
template <typename T, size_t N>
inline PackedTensorAccessor32<T, N> packed_accessor32(torch::stable::Tensor t) {
  return PackedTensorAccessor32<T, N>(
      static_cast<typename PackedTensorAccessor32<T, N>::PtrType>(t.data_ptr()),
      t.sizes().data(),
      t.strides().data());
}

template <typename T, size_t N>
using PackedTensorAccessorSizeT =
    torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<
        T,
        N,
        torch::headeronly::RestrictPtrTraits,
        size_t>;

template <typename T, size_t N>
inline PackedTensorAccessorSizeT<T, N> packed_accessor_size_t(
    torch::stable::Tensor t) {
  return PackedTensorAccessorSizeT<T, N>(
      static_cast<typename PackedTensorAccessorSizeT<T, N>::PtrType>(
          t.data_ptr()),
      t.sizes().data(),
      t.strides().data());
}

#endif

} // namespace torchaudio
