#include <c10/core/ScalarType.h>
#include <sox.h>
#include <torchaudio/csrc/sox/utils.h>

namespace torchaudio {
namespace sox_utils {

void set_seed(const int64_t seed) {
  sox_get_globals()->ranqd1 = static_cast<sox_int32_t>(seed);
}

void set_verbosity(const int64_t verbosity) {
  sox_get_globals()->verbosity = static_cast<unsigned>(verbosity);
}

void set_use_threads(const bool use_threads) {
  sox_get_globals()->use_threads = static_cast<sox_bool>(use_threads);
}

void set_buffer_size(const int64_t buffer_size) {
  sox_get_globals()->bufsiz = static_cast<size_t>(buffer_size);
}

std::vector<std::vector<std::string>> list_effects() {
  std::vector<std::vector<std::string>> effects;
  for (const sox_effect_fn_t* fns = sox_get_effect_fns(); *fns; ++fns) {
    const sox_effect_handler_t* handler = (*fns)();
    if (handler && handler->name) {
      if (UNSUPPORTED_EFFECTS.find(handler->name) ==
          UNSUPPORTED_EFFECTS.end()) {
        effects.emplace_back(std::vector<std::string>{
            handler->name,
            handler->usage ? std::string(handler->usage) : std::string("")});
      }
    }
  }
  return effects;
}

std::vector<std::string> list_write_formats() {
  std::vector<std::string> formats;
  for (const sox_format_tab_t* fns = sox_get_format_fns(); fns->fn; ++fns) {
    const sox_format_handler_t* handler = fns->fn();
    for (const char* const* names = handler->names; *names; ++names) {
      if (!strchr(*names, '/') && handler->write)
        formats.emplace_back(*names);
    }
  }
  return formats;
}

std::vector<std::string> list_read_formats() {
  std::vector<std::string> formats;
  for (const sox_format_tab_t* fns = sox_get_format_fns(); fns->fn; ++fns) {
    const sox_format_handler_t* handler = fns->fn();
    for (const char* const* names = handler->names; *names; ++names) {
      if (!strchr(*names, '/') && handler->read)
        formats.emplace_back(*names);
    }
  }
  return formats;
}

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
  close();
}

sox_format_t* SoxFormat::operator->() const noexcept {
  return fd_;
}
SoxFormat::operator sox_format_t*() const noexcept {
  return fd_;
}

void SoxFormat::close() {
  if (fd_ != nullptr) {
    sox_close(fd_);
    fd_ = nullptr;
  }
}

void validate_input_file(const SoxFormat& sf, bool check_length) {
  if (static_cast<sox_format_t*>(sf) == nullptr) {
    throw std::runtime_error("Error loading audio file: failed to open file.");
  }
  if (sf->encoding.encoding == SOX_ENCODING_UNKNOWN) {
    throw std::runtime_error("Error loading audio file: unknown encoding.");
  }
  if (check_length && sf->signal.length == 0) {
    throw std::runtime_error("Error reading audio file: unknown length.");
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
  if (filetype == "wav" || filetype == "amb") {
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
  if (filetype == "sph")
    return SOX_ENCODING_SIGN2;
  if (filetype == "amr-nb")
    return SOX_ENCODING_AMR_NB;
  throw std::runtime_error("Unsupported file type: " + filetype);
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
  if (filetype == "wav" || filetype == "amb") {
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
  if (filetype == "sph")
    return 32;
  if (filetype == "amr-nb") {
    TORCH_INTERNAL_ASSERT(
        dtype == torch::kInt16,
        "When saving to AMR-NB format, the input tensor must be int16 type.");
    return 16;
  }
  throw std::runtime_error("Unsupported file type: " + filetype);
}

sox_signalinfo_t get_signalinfo(
    const TensorSignal* signal,
    const std::string filetype) {
  auto tensor = signal->getTensor();
  return sox_signalinfo_t{
      /*rate=*/static_cast<sox_rate_t>(signal->getSampleRate()),
      /*channels=*/
      static_cast<unsigned>(tensor.size(signal->getChannelsFirst() ? 0 : 1)),
      /*precision=*/get_precision(filetype, tensor.dtype()),
      /*length=*/static_cast<uint64_t>(tensor.numel())};
}

sox_encodinginfo_t get_encodinginfo(
    const std::string filetype,
    const caffe2::TypeMeta dtype) {
  return sox_encodinginfo_t{
      /*encoding=*/get_encoding(filetype, dtype),
      /*bits_per_sample=*/get_precision(filetype, dtype),
      /*compression=*/HUGE_VAL,
      /*reverse_bytes=*/sox_option_default,
      /*reverse_nibbles=*/sox_option_default,
      /*reverse_bits=*/sox_option_default,
      /*opposite_endian=*/sox_false};
}

sox_encodinginfo_t get_encodinginfo(
    const std::string filetype,
    const caffe2::TypeMeta dtype,
    c10::optional<double>& compression) {
  return sox_encodinginfo_t{
      /*encoding=*/get_encoding(filetype, dtype),
      /*bits_per_sample=*/get_precision(filetype, dtype),
      /*compression=*/compression.value_or(HUGE_VAL),
      /*reverse_bytes=*/sox_option_default,
      /*reverse_nibbles=*/sox_option_default,
      /*reverse_bits=*/sox_option_default,
      /*opposite_endian=*/sox_false};
}

#ifdef TORCH_API_INCLUDE_EXTENSION_H

uint64_t read_fileobj(py::object* fileobj, const uint64_t size, char* buffer) {
  uint64_t num_read = 0;
  while (num_read < size) {
    auto request = size - num_read;
    auto chunk = static_cast<std::string>(
        static_cast<py::bytes>(fileobj->attr("read")(request)));
    auto chunk_len = chunk.length();
    if (chunk_len == 0) {
      break;
    }
    if (chunk_len > request) {
      std::ostringstream message;
      message
          << "Requested up to " << request << " bytes but, "
          << "received " << chunk_len << " bytes. "
          << "The given object does not confirm to read protocol of file object.";
      throw std::runtime_error(message.str());
    }
    memcpy(buffer, chunk.data(), chunk_len);
    buffer += chunk_len;
    num_read += chunk_len;
  }
  return num_read;
}

#endif // TORCH_API_INCLUDE_EXTENSION_H

} // namespace sox_utils
} // namespace torchaudio
