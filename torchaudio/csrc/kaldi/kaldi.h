#ifndef TORCHAUDIO_CSRC_KALDI_WRAPPER_H
#define TORCHAUDIO_CSRC_KALDI_WRAPPER_H

#include <torch/script.h>

namespace torchaudio {
namespace kaldi {

  torch::Tensor ComputeKaldiPitch(
      const torch::Tensor &wave,
      double sample_frequency,
      double frame_length,
      double frame_shift,
      double preemphasis_coefficient,
      double min_f0,
      double max_f0,
      double soft_min_f0,
      double penalty_factor,
      double lowpass_cutoff,
      double resample_frequency,
      double delta_pitch,
      double nccf_ballast,
      int64_t lowpass_filter_width,
      int64_t upsample_filter_width,
      int64_t max_frames_latency,
      int64_t frames_per_chunk,
      bool simulate_first_pass_online,
      int64_t recompute_frame,
      bool nccf_ballast_online,
      bool snip_edges);

} // namespace kaldi
} // namespace torchaudio

#endif
