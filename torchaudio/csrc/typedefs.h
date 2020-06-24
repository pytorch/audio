#ifndef TORCHAUDIO_TYPDEFS_H
#define TORCHAUDIO_TYPDEFS_H

#include <torch/script.h>

namespace torchaudio {
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

} // namespace torchaudio

#endif
