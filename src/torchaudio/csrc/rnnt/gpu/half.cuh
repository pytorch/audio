#pragma once

#ifdef USE_C10_HALF
#include "c10/util/Half.h"
#endif // USE_C10_HALF

#include <libtorchaudio/rnnt/macros.h>

namespace torchaudio {
namespace rnnt {

struct alignas(sizeof(__half)) Half {
  __half x;

  HOST_AND_DEVICE Half() = default;

  FORCE_INLINE HOST_AND_DEVICE Half(float f) {
    x = __float2half_rn(f);
    if (isinf(__half2float(x))) {
      x = __float2half_rz(f); // round toward 0.
    }
  }

  FORCE_INLINE HOST_AND_DEVICE operator float() const {
    return __half2float(x);
  }

  FORCE_INLINE HOST_AND_DEVICE Half(__half f) {
    x = f;
  }

  FORCE_INLINE HOST_AND_DEVICE operator __half() const {
    return x;
  }
};

} // namespace rnnt
} // namespace torchaudio
