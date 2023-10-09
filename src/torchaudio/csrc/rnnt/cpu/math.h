#pragma once

#include <libtorchaudio/rnnt/macros.h>

namespace torchaudio {
namespace rnnt {

namespace math {

template <typename DTYPE>
FORCE_INLINE HOST_AND_DEVICE DTYPE max(DTYPE x, DTYPE y) {
  if (x > y)
    return x;
  else
    return y;
}

template <typename DTYPE>
FORCE_INLINE HOST_AND_DEVICE DTYPE min(DTYPE x, DTYPE y) {
  if (x > y)
    return y;
  else
    return x;
}

// log_sum_exp
template <typename DTYPE>
FORCE_INLINE HOST_AND_DEVICE DTYPE lse(DTYPE x, DTYPE y);

template <>
FORCE_INLINE HOST_AND_DEVICE float lse(float x, float y) {
  if (y > x) {
    return y + log1pf(expf(x - y));
  } else {
    return x + log1pf(expf(y - x));
  }
}

} // namespace math

} // namespace rnnt
} // namespace torchaudio
