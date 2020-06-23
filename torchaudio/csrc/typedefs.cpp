#include <torchaudio/csrc/typedefs.h>

namespace torchaudio {
SignalInfo::SignalInfo(
    const int64_t sample_rate_,
    const int64_t num_channels_,
    const int64_t num_frames_)
    : sample_rate(sample_rate_),
      num_channels(num_channels_),
      num_frames(num_frames_){};

int64_t SignalInfo::getSampleRate() const {
  return sample_rate;
}

int64_t SignalInfo::getNumChannels() const {
  return num_channels;
}

int64_t SignalInfo::getNumFrames() const {
  return num_frames;
}
} // namespace torchaudio
