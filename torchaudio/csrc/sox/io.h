#ifndef TORCHAUDIO_SOX_IO_H
#define TORCHAUDIO_SOX_IO_H

#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>
#endif // TORCH_API_INCLUDE_EXTENSION_H

#include <torch/script.h>
#include <torchaudio/csrc/sox/utils.h>

namespace torchaudio {
namespace sox_io {

struct SignalInfo : torch::CustomClassHolder {
  int64_t sample_rate;
  int64_t num_channels;
  int64_t num_frames;

  SignalInfo(
      const int64_t sample_rate_,
      const int64_t num_channels_,
      const int64_t num_frames_);
  int64_t getSampleRate() const;
  int64_t getNumChannels() const;
  int64_t getNumFrames() const;
};

c10::intrusive_ptr<SignalInfo> get_info(
    const std::string& path,
    c10::optional<std::string>& format);

c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal> load_audio_file(
    const std::string& path,
    c10::optional<int64_t>& frame_offset,
    c10::optional<int64_t>& num_frames,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
    c10::optional<std::string>& format);

void save_audio_file(
    const std::string& file_name,
    const c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal>& signal,
    const double compression = 0.);

#ifdef TORCH_API_INCLUDE_EXTENSION_H

std::tuple<torch::Tensor, int64_t> load_audio_fileobj(
    py::object fileobj,
    c10::optional<int64_t>& frame_offset,
    c10::optional<int64_t>& num_frames,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
    c10::optional<std::string>& format);

#endif // TORCH_API_INCLUDE_EXTENSION_H

} // namespace sox_io
} // namespace torchaudio

#endif
