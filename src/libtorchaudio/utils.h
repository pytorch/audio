#pragma once
#include <torch/types.h>

namespace torchaudio {
bool is_rir_available();
bool is_align_available();
std::optional<int64_t> cuda_version();
} // namespace torchaudio
