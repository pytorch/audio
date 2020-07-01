#include <sox.h>
#include <torchaudio/csrc/sox_io.h>
#include <torchaudio/csrc/sox_utils.h>

using namespace torch::indexing;
using namespace torchaudio::sox_utils;

namespace torchaudio {
namespace sox_io {

c10::intrusive_ptr<torchaudio::SignalInfo> get_info(const std::string& path) {
  SoxFormat sf(sox_open_read(
      path.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));

  if (static_cast<sox_format_t*>(sf) == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }

  return c10::make_intrusive<torchaudio::SignalInfo>(
      static_cast<int64_t>(sf->signal.rate),
      static_cast<int64_t>(sf->signal.channels),
      static_cast<int64_t>(sf->signal.length / sf->signal.channels));
}

c10::intrusive_ptr<TensorSignal> load_audio_file(
    const std::string& path,
    const int64_t frame_offset,
    const int64_t num_frames,
    const bool normalize,
    const bool channels_first) {
  if (frame_offset < 0) {
    throw std::runtime_error(
        "Invalid argument: frame_offset must be non-negative.");
  }
  if (num_frames == 0 || num_frames < -1) {
    throw std::runtime_error(
        "Invalid argument: num_frames must be -1 or greater than 0.");
  }

  SoxFormat sf(sox_open_read(
      path.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));

  validate_input_file(sf);

  const int64_t num_channels = sf->signal.channels;
  const int64_t num_total_samples = sf->signal.length;
  const int64_t sample_start = sf->signal.channels * frame_offset;

  if (sox_seek(sf, sample_start, 0) == SOX_EOF) {
    throw std::runtime_error("Error reading audio file: offset past EOF.");
  }

  const int64_t sample_end = [&]() {
    if (num_frames == -1)
      return num_total_samples;
    const int64_t sample_end_ = num_channels * num_frames + sample_start;
    if (num_total_samples < sample_end_) {
      // For lossy encoding, it is difficult to predict exact size of buffer for
      // reading the number of samples required.
      // So we allocate buffer size of given `num_frames` and ask sox to read as
      // much as possible. For lossless format, sox reads exact number of
      // samples, but for lossy encoding, sox can end up reading less. (i.e.
      // mp3) For the consistent behavior specification between lossy/lossless
      // format, we allow users to provide `num_frames` value that exceeds #of
      // available samples, and we adjust it here.
      return num_total_samples;
    }
    return sample_end_;
  }();

  const int64_t max_samples = sample_end - sample_start;

  // Read samples into buffer
  std::vector<sox_sample_t> buffer;
  buffer.reserve(max_samples);
  const int64_t num_samples = sox_read(sf, buffer.data(), max_samples);
  if (num_samples == 0) {
    throw std::runtime_error(
        "Error reading audio file: empty file or read operation failed.");
  }
  // NOTE: num_samples may be smaller than max_samples if the input
  // format is compressed (i.e. mp3).

  // Convert to Tensor
  auto tensor = convert_to_tensor(
      buffer.data(),
      num_samples,
      num_channels,
      get_dtype(sf->encoding.encoding, sf->signal.precision),
      normalize,
      channels_first);

  return c10::make_intrusive<TensorSignal>(
      tensor, static_cast<int64_t>(sf->signal.rate), channels_first);
}

void save_audio_file(
    const std::string& file_name,
    const c10::intrusive_ptr<TensorSignal>& signal,
    const double compression,
    const int64_t frames_per_chunk) {
  const auto tensor = signal->getTensor();
  const auto sample_rate = signal->getSampleRate();
  const auto channels_first = signal->getChannelsFirst();

  validate_input_tensor(tensor);

  const auto filetype = get_filetype(file_name);
  const auto signal_info =
      get_signalinfo(tensor, sample_rate, channels_first, filetype);
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

  auto tensor_ = tensor;
  if (channels_first) {
    tensor_ = tensor_.t();
  }

  for (int64_t i = 0; i < tensor_.size(0); i += frames_per_chunk) {
    auto chunk = tensor_.index({Slice(i, i + frames_per_chunk), Slice()});
    chunk = unnormalize_wav(chunk).contiguous();

    const size_t numel = chunk.numel();
    if (sox_write(sf, chunk.data_ptr<int32_t>(), numel) != numel) {
      throw std::runtime_error(
          "Error saving audio file: failed to write the entier buffer.");
    }
  }
}

} // namespace sox_io
} // namespace torchaudio
