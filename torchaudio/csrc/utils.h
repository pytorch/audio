#pragma once
#include <torch/torch.h>

namespace torchaudio {
bool is_kaldi_available();
bool is_rir_available();
c10::optional<int64_t> cuda_version();
} // namespace torchaudio
