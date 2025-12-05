#pragma once

#include <torch/headeronly/core/TensorAccessor.h>

// TODO: replace the include libtorchaudio/stable/ops.h with
// torch/stable/ops.h when torch::stable provides all required
// features (torch::stable::item<T> et al):
#include <libtorchaudio/stable/ops.h>

namespace torchaudio {

namespace util {
template <typename T>
T max(const torch::stable::Tensor& t) {
  return torchaudio::stable::item<T>(torch::stable::amax(t, {}));
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
