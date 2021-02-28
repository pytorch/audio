#include <torch/script.h>
#include "kaldifeat/csrc/feature-fbank.h"

namespace torchaudio {
namespace kaldifeat {


torch::Tensor compute_fbank(
    const torch::Tensor &wave,
    int64_t sample_rate) {
  ::kaldifeat::FbankOptions option;
  option.frame_opts.samp_freq = static_cast<float>(sample_rate);
  option.device = wave.device();
  ::kaldifeat::Fbank fbank(option);
  float vtln_warp = 1.0f;
  torch::Tensor feats = fbank.ComputeFeatures(wave, vtln_warp);
  return feats;
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
    "torchaudio::kaldifeat_compute_fbank",
    &torchaudio::kaldifeat::compute_fbank);
}

} // kaldifeat
} // torchaudio
