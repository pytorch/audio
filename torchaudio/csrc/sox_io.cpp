#include <sox.h>
#include <torchaudio/csrc/sox_effects.h>
#include <torchaudio/csrc/sox_effects_chain.h>
#include <torchaudio/csrc/sox_io.h>
#include <torchaudio/csrc/sox_utils.h>

using namespace torch::indexing;
using namespace torchaudio::sox_utils;

namespace torchaudio {
namespace sox_io {

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

c10::intrusive_ptr<SignalInfo> get_info(const std::string& path) {
  SoxFormat sf(sox_open_read(
      path.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));

  if (static_cast<sox_format_t*>(sf) == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }

  return c10::make_intrusive<SignalInfo>(
      static_cast<int64_t>(sf->signal.rate),
      static_cast<int64_t>(sf->signal.channels),
      static_cast<int64_t>(sf->signal.length / sf->signal.channels));
}

c10::intrusive_ptr<TensorSignal> load_audio_file(
    const std::string& path,
    const int64_t frame_offset,
    const int64_t num_frames,
    const bool normalize,
    const bool channels_first,
    const int64_t sample_rate) {
  if (frame_offset < 0) {
    throw std::runtime_error(
        "Invalid argument: frame_offset must be non-negative.");
  }
  if (num_frames == 0 || num_frames < -1) {
    throw std::runtime_error(
        "Invalid argument: num_frames must be -1 or greater than 0.");
  }
  if (sample_rate == 0 || sample_rate < -1) {
    throw std::runtime_error(
        "Invalid argument: sample_rate must be -1 or greater than 0.");
  }

  std::vector<std::vector<std::string>> effects;
  if (sample_rate != -1) {
    effects.emplace_back(
        std::vector<std::string>{"rate", std::to_string(sample_rate)});
  }
  if (num_frames != -1) {
    std::ostringstream offset, frames;
    offset << frame_offset << "s";
    frames << "+" << num_frames << "s";
    effects.emplace_back(
        std::vector<std::string>{"trim", offset.str(), frames.str()});
  } else if (frame_offset != 0) {
    std::ostringstream offset;
    offset << frame_offset << "s";
    effects.emplace_back(std::vector<std::string>{"trim", offset.str()});
  }

  return torchaudio::sox_effects::apply_effects_file(
      path, effects, normalize, channels_first);
}

void save_audio_file(
    const std::string& file_name,
    const c10::intrusive_ptr<TensorSignal>& signal,
    const double compression) {
  const auto tensor = signal->getTensor();

  validate_input_tensor(tensor);

  const auto filetype = get_filetype(file_name);
  const auto signal_info = get_signalinfo(signal.get(), filetype);
  const auto encoding_info =
      get_encodinginfo(filetype, tensor.dtype(), compression);

  SoxFormat sf(sox_open_write(
      file_name.c_str(),
      &signal_info,
      &encoding_info,
      /*filetype=*/filetype.c_str(),
      /*oob=*/nullptr,
      /*overwrite_permitted=*/nullptr));

  if (static_cast<sox_format_t*>(sf) == nullptr) {
    throw std::runtime_error("Error saving audio file: failed to open file.");
  }

  torchaudio::sox_effects_chain::SoxEffectsChain chain(
      /*input_encoding=*/get_encodinginfo("wav", tensor.dtype(), 0.),
      /*output_encoding=*/sf->encoding);
  chain.addInputTensor(signal.get());
  chain.addOutputFile(sf);
  chain.run();
}

} // namespace sox_io
} // namespace torchaudio
