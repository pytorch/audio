#include <torch/script.h>
#include "feat/pitch-functions.h"

namespace torchaudio {
namespace kaldi {

namespace {

torch::Tensor denormalize(const torch::Tensor& t) {
  auto ret = t;
  auto pos = t > 0, neg = t < 0;
  ret.index_put({pos}, t.index({pos}) * 32767);
  ret.index_put({neg}, t.index({neg}) * 32768);
  return ret;
}

torch::Tensor compute_kaldi_pitch(
    const torch::Tensor& wave,
    const ::kaldi::PitchExtractionOptions& opts) {
  ::kaldi::VectorBase<::kaldi::BaseFloat> input(wave);
  ::kaldi::Matrix<::kaldi::BaseFloat> output;
  ::kaldi::ComputeKaldiPitch(opts, input, &output);
  return output.tensor_;
}

} // namespace

torch::Tensor ComputeKaldiPitch(
    const torch::Tensor& wave,
    double sample_frequency,
    double frame_length,
    double frame_shift,
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
    bool snip_edges) {
  TORCH_CHECK(wave.ndimension() == 2, "Input tensor must be 2 dimentional.");
  TORCH_CHECK(wave.device().is_cpu(), "Input tensor must be on CPU.");
  TORCH_CHECK(
      wave.dtype() == torch::kFloat32, "Input tensor must be float32 type.");

  ::kaldi::PitchExtractionOptions opts;
  opts.samp_freq = static_cast<::kaldi::BaseFloat>(sample_frequency);
  opts.frame_shift_ms = static_cast<::kaldi::BaseFloat>(frame_shift);
  opts.frame_length_ms = static_cast<::kaldi::BaseFloat>(frame_length);
  opts.min_f0 = static_cast<::kaldi::BaseFloat>(min_f0);
  opts.max_f0 = static_cast<::kaldi::BaseFloat>(max_f0);
  opts.soft_min_f0 = static_cast<::kaldi::BaseFloat>(soft_min_f0);
  opts.penalty_factor = static_cast<::kaldi::BaseFloat>(penalty_factor);
  opts.lowpass_cutoff = static_cast<::kaldi::BaseFloat>(lowpass_cutoff);
  opts.resample_freq = static_cast<::kaldi::BaseFloat>(resample_frequency);
  opts.delta_pitch = static_cast<::kaldi::BaseFloat>(delta_pitch);
  opts.lowpass_filter_width = static_cast<::kaldi::int32>(lowpass_filter_width);
  opts.upsample_filter_width =
      static_cast<::kaldi::int32>(upsample_filter_width);
  opts.max_frames_latency = static_cast<::kaldi::int32>(max_frames_latency);
  opts.frames_per_chunk = static_cast<::kaldi::int32>(frames_per_chunk);
  opts.simulate_first_pass_online = simulate_first_pass_online;
  opts.recompute_frame = static_cast<::kaldi::int32>(recompute_frame);
  opts.snip_edges = snip_edges;

  // Kaldi's float type expects value range of int16 expressed as float
  torch::Tensor wave_ = denormalize(wave);

  auto batch_size = wave_.size(0);
  std::vector<torch::Tensor> results(batch_size);
  at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
    for (auto i = begin; i < end; ++i) {
      results[i] = compute_kaldi_pitch(wave_.index({i}), opts);
    }
  });
  return torch::stack(results, 0);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "torchaudio::kaldi_ComputeKaldiPitch",
      &torchaudio::kaldi::ComputeKaldiPitch);
}

} // namespace kaldi
} // namespace torchaudio
