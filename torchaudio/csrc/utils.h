#pragma once
#include <torch/types.h>

namespace torchaudio {
bool is_rir_available();
bool is_align_available();
c10::optional<int64_t> cuda_version();
int find_avutil(const char* name);
} // namespace torchaudio
