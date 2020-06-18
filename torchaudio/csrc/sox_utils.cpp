#include <c10/core/ScalarType.h>
#include <sox.h>
#include <torchaudio/csrc/sox_utils.h>

namespace torchaudio {
namespace sox_utils {

TensorSignal::TensorSignal(
    torch::Tensor tensor_,
    int64_t sample_rate_,
    bool channels_first_)
    : tensor(tensor_),
      sample_rate(sample_rate_),
      channels_first(channels_first_){};

torch::Tensor TensorSignal::getTensor() const {
  return tensor;
}
int64_t TensorSignal::getSampleRate() const {
  return sample_rate;
}
bool TensorSignal::getChannelsFirst() const {
  return channels_first;
}

SoxFormat::SoxFormat(sox_format_t* fd) noexcept : fd_(fd) {}
SoxFormat::~SoxFormat() {
  if (fd_ != nullptr) {
    sox_close(fd_);
  }
}
sox_format_t* SoxFormat::operator->() const noexcept {
  return fd_;
}
sox_format_t* SoxFormat::get() const noexcept {
  return fd_;
}

void validate_input_file(const SoxFormat& sf) {
  if (sf.get() == nullptr) {
    throw std::runtime_error("Error loading audio file: failed to open file.");
  }
  if (sf->encoding.encoding == SOX_ENCODING_UNKNOWN) {
    throw std::runtime_error("Error loading audio file: unknown encoding.");
  }
  if (sf->signal.length == 0) {
    throw std::runtime_error("Error reading audio file: unkown length.");
  }
}

caffe2::TypeMeta get_dtype(
    const sox_encoding_t encoding,
    const unsigned precision) {
  const auto dtype = [&]() {
    switch (encoding) {
      case SOX_ENCODING_UNSIGNED: // 8-bit PCM WAV
        return torch::kUInt8;
      case SOX_ENCODING_SIGN2: // 16-bit or 32-bit PCM WAV
        switch (precision) {
          case 16:
            return torch::kInt16;
          case 32:
            return torch::kInt32;
          default:
            throw std::runtime_error(
                "Only 16 and 32 bits are supported for signed PCM.");
        }
      default:
        // default to float32 for the other formats, including
        // 32-bit flaoting-point WAV,
        // MP3,
        // FLAC,
        // VORBIS etc...
        return torch::kFloat32;
    }
  }();
  return c10::scalarTypeToTypeMeta(dtype);
}

torch::Tensor convert_to_tensor(
    sox_sample_t* buffer,
    const int32_t num_samples,
    const int32_t num_channels,
    const caffe2::TypeMeta dtype,
    const bool normalize,
    const bool channels_first) {
  auto t = torch::from_blob(
      buffer, {num_samples / num_channels, num_channels}, torch::kInt32);
  // Note: Tensor created from_blob does not own data but borrwos
  // So make sure to create a new copy after processing samples.
  if (normalize || dtype == torch::kFloat32) {
    t = t.to(torch::kFloat32);
    t *= (t > 0) / 2147483647. + (t < 0) / 2147483648.;
  } else if (dtype == torch::kInt32) {
    t = t.clone();
  } else if (dtype == torch::kInt16) {
    t.floor_divide_(1 << 16);
    t = t.to(torch::kInt16);
  } else if (dtype == torch::kUInt8) {
    t.floor_divide_(1 << 24);
    t += 128;
    t = t.to(torch::kUInt8);
  } else {
    throw std::runtime_error("Unsupported dtype.");
  }
  if (channels_first) {
    t = t.transpose(1, 0);
  }
  return t.contiguous();
}

} // namespace sox_utils
} // namespace torchaudio
