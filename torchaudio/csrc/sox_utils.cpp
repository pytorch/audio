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
SoxFormat::operator sox_format_t*() const noexcept {
  return fd_;
}

void validate_input_file(const SoxFormat& sf) {
  if (static_cast<sox_format_t*>(sf) == nullptr) {
    throw std::runtime_error("Error loading audio file: failed to open file.");
  }
  if (sf->encoding.encoding == SOX_ENCODING_UNKNOWN) {
    throw std::runtime_error("Error loading audio file: unknown encoding.");
  }
  if (sf->signal.length == 0) {
    throw std::runtime_error("Error reading audio file: unkown length.");
  }
}

void validate_input_tensor(const torch::Tensor tensor) {
  if (!tensor.device().is_cpu()) {
    throw std::runtime_error("Input tensor has to be on CPU.");
  }

  if (tensor.ndimension() != 2) {
    throw std::runtime_error("Input tensor has to be 2D.");
  }

  const auto dtype = tensor.dtype();
  if (!(dtype == torch::kFloat32 || dtype == torch::kInt32 ||
        dtype == torch::kInt16 || dtype == torch::kUInt8)) {
    throw std::runtime_error(
        "Input tensor has to be one of float32, int32, int16 or uint8 type.");
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

torch::Tensor unnormalize_wav(const torch::Tensor input_tensor) {
  const auto dtype = input_tensor.dtype();
  auto tensor = input_tensor;
  if (dtype == torch::kFloat32) {
    double multi_pos = 2147483647.;
    double multi_neg = -2147483648.;
    auto mult = (tensor > 0) * multi_pos - (tensor < 0) * multi_neg;
    tensor = tensor.to(torch::dtype(torch::kFloat64));
    tensor *= mult;
    tensor.clamp_(multi_neg, multi_pos);
    tensor = tensor.to(torch::dtype(torch::kInt32));
  } else if (dtype == torch::kInt32) {
    // already denormalized
  } else if (dtype == torch::kInt16) {
    tensor = tensor.to(torch::dtype(torch::kInt32));
    tensor *= ((tensor != 0) * 65536);
  } else if (dtype == torch::kUInt8) {
    tensor = tensor.to(torch::dtype(torch::kInt32));
    tensor -= 128;
    tensor *= 16777216;
  } else {
    throw std::runtime_error("Unexpected dtype.");
  }
  return tensor;
}

const std::string get_filetype(const std::string path) {
  std::string ext = path.substr(path.find_last_of(".") + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext;
}

sox_encoding_t get_encoding(
    const std::string filetype,
    const caffe2::TypeMeta dtype) {
  if (filetype == "mp3")
    return SOX_ENCODING_MP3;
  if (filetype == "flac")
    return SOX_ENCODING_FLAC;
  if (filetype == "ogg" || filetype == "vorbis")
    return SOX_ENCODING_VORBIS;
  if (filetype == "wav") {
    if (dtype == torch::kUInt8)
      return SOX_ENCODING_UNSIGNED;
    if (dtype == torch::kInt16)
      return SOX_ENCODING_SIGN2;
    if (dtype == torch::kInt32)
      return SOX_ENCODING_SIGN2;
    if (dtype == torch::kFloat32)
      return SOX_ENCODING_FLOAT;
    throw std::runtime_error("Unsupported dtype.");
  }
  throw std::runtime_error("Unsupported file type.");
}

unsigned get_precision(
    const std::string filetype,
    const caffe2::TypeMeta dtype) {
  if (filetype == "mp3")
    return SOX_UNSPEC;
  if (filetype == "flac")
    return 24;
  if (filetype == "ogg" || filetype == "vorbis")
    return SOX_UNSPEC;
  if (filetype == "wav") {
    if (dtype == torch::kUInt8)
      return 8;
    if (dtype == torch::kInt16)
      return 16;
    if (dtype == torch::kInt32)
      return 32;
    if (dtype == torch::kFloat32)
      return 32;
    throw std::runtime_error("Unsupported dtype.");
  }
  throw std::runtime_error("Unsupported file type.");
}

sox_signalinfo_t get_signalinfo(
    const torch::Tensor& tensor,
    const int64_t sample_rate,
    const bool channels_first,
    const std::string filetype) {
  return sox_signalinfo_t{
      /*rate=*/static_cast<sox_rate_t>(sample_rate),
      /*channels=*/static_cast<unsigned>(tensor.size(channels_first ? 0 : 1)),
      /*precision=*/get_precision(filetype, tensor.dtype()),
      /*length=*/static_cast<uint64_t>(tensor.numel())};
}

sox_encodinginfo_t get_encodinginfo(
    const std::string filetype,
    const caffe2::TypeMeta dtype,
    const double compression) {
  const double compression_ = [&]() {
    if (filetype == "mp3")
      return compression;
    if (filetype == "flac")
      return compression;
    if (filetype == "ogg" || filetype == "vorbis")
      return compression;
    if (filetype == "wav")
      return 0.;
    throw std::runtime_error("Unsupported file type.");
  }();

  return sox_encodinginfo_t{/*encoding=*/get_encoding(filetype, dtype),
                            /*bits_per_sample=*/get_precision(filetype, dtype),
                            /*compression=*/compression_,
                            /*reverse_bytes=*/sox_option_default,
                            /*reverse_nibbles=*/sox_option_default,
                            /*reverse_bits=*/sox_option_default,
                            /*opposite_endian=*/sox_false};
}

} // namespace sox_utils
} // namespace torchaudio
