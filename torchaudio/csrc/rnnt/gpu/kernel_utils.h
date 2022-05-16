#pragma once

#include <cassert>

#ifdef __HIP_PLATFORM_AMD__
#include <torchaudio/csrc/rnnt/hip/math_hip.cuh>
#else
#include <torchaudio/csrc/rnnt/gpu/math.cuh>
#endif

namespace torchaudio {
namespace rnnt {

inline HOST_AND_DEVICE bool in_range(
    int start,
    int end, // inclusive
    int val) {
  return start <= val && val <= end;
}

#define LOG_PROBS_SKIP_IDX 0
#define LOG_PROBS_EMIT_IDX 1

struct Indexer2D {
  const int& size2_;

  FORCE_INLINE HOST_AND_DEVICE Indexer2D(const int& size2) : size2_(size2) {}

  FORCE_INLINE HOST_AND_DEVICE int operator()(int index1, int index2) {
    return index1 * size2_ + index2;
  }
};

struct Indexer3D {
  const int& size2_;
  const int& size3_;

  FORCE_INLINE HOST_AND_DEVICE Indexer3D(const int& size2, const int& size3)
      : size2_(size2), size3_(size3) {}

  FORCE_INLINE HOST_AND_DEVICE int operator()(
      int index1,
      int index2,
      int index3) {
    return (index1 * size2_ + index2) * size3_ + index3;
  }
};

struct Indexer4D {
  const int& size2_;
  const int& size3_;
  const int& size4_;

  HOST_AND_DEVICE Indexer4D(
      const int& size2,
      const int& size3,
      const int& size4)
      : size2_(size2), size3_(size3), size4_(size4) {}

  HOST_AND_DEVICE int operator()(
      int index1,
      int index2,
      int index3,
      int index4) {
    return ((index1 * size2_ + index2) * size3_ + index3) * size4_ + index4;
  }
};

} // namespace rnnt
} // namespace torchaudio
