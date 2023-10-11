#include <libtorio/ffmpeg/stream_reader/stream_reader.h>
#include <torch/script.h>
#include <stdexcept>

namespace torchaudio {
namespace io {
namespace {

torch::Tensor _load_audio(
    StreamReader& s,
    int i,
    const c10::optional<std::string>& filter,
    const bool& channels_first) {
  s.add_audio_stream(i, -1, -1, filter, {}, {});
  s.process_all_packets();
  auto chunk = s.pop_chunks()[0];
  TORCH_CHECK(chunk, "Failed to decode audio.");
  auto waveform = chunk.value().frames;
  return channels_first ? waveform.transpose(0, 1) : waveform;
}

std::tuple<torch::Tensor, int64_t> load(
    const std::string& src,
    const c10::optional<std::string>& format,
    const c10::optional<std::string>& filter,
    const bool& channels_first) {
  StreamReader s{src, format, {}};
  auto i = s.find_best_audio_stream();
  auto sample_rate = s.get_src_stream_info(i).sample_rate;
  auto waveform = _load_audio(s, i, filter, channels_first);
  return {waveform, sample_rate};
}

std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> info(
    const std::string& src,
    const c10::optional<std::string>& format) {
  StreamReader s{src, format, {}};
  auto i = s.find_best_audio_stream();
  auto sinfo = s.get_src_stream_info(i);
  int64_t num_frames = [&]() {
    if (sinfo.num_frames == 0) {
      torch::Tensor waveform = _load_audio(s, i, {}, false);
      return waveform.size(0);
    }
    return sinfo.num_frames;
  }();
  return {
      static_cast<int64_t>(sinfo.sample_rate),
      static_cast<int64_t>(num_frames),
      static_cast<int64_t>(sinfo.num_channels),
      static_cast<int64_t>(sinfo.bits_per_sample),
      sinfo.codec_name};
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::compat_load", &load);
  m.def("torchaudio::compat_info", &info);
}

} // namespace
} // namespace io
} // namespace torchaudio
