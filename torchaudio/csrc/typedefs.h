#ifndef TORCHAUDIO_TYPDEFS_H
#define TORCHAUDIO_TYPDEFS_H

#include <torch/script.h>

namespace torchaudio {
struct SignalInfo : torch::CustomClassHolder {
  int64_t sample_rate;
  int64_t num_channels;
  int64_t num_samples;

  SignalInfo(
      const int64_t sample_rate_,
      const int64_t num_channels_,
      const int64_t num_samples_);
  int64_t getSampleRate() const;
  int64_t getNumChannels() const;
  int64_t getNumSamples() const;
};

} // namespace torchaudio

#endif
