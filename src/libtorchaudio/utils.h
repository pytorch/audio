#pragma once

// TODO: replace the include libtorchaudio/stable/ops.h with
// torch/stable/ops.h when torch::stable provides all required
// features (torch::stable::item<T> or similar):
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

} // namespace torchaudio
