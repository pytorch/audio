#ifndef TORCHAUDIO_SOX_IO_H
#define TORCHAUDIO_SOX_IO_H

#include <torch/script.h>
#include <torchaudio/csrc/sox_utils.h>

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

c10::intrusive_ptr<SignalInfo> get_info(const std::string& path);

// ver. 0
c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal> load_audio_file(
    const std::string& path,
    const int64_t frame_offset = 0,
    const int64_t num_frames = -1,
    const bool normalize = true,
    const bool channels_first = true);

// ver. 1 sample_rate is added
c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal> load_audio_file_v1(
    const std::string& path,
    const int64_t frame_offset = 0,
    const int64_t num_frames = -1,
    const bool normalize = true,
    const bool channels_first = true,
    const int64_t sample_rate = -1);

void save_audio_file(
    const std::string& file_name,
    const c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal>& signal,
    const double compression = 0.);

} // namespace sox_io
} // namespace torchaudio

#endif
