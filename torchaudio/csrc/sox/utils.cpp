#include <c10/core/ScalarType.h>
#include <sox.h>
#include <torchaudio/csrc/sox/types.h>
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

caffe2::TypeMeta get_dtype_from_str(const std::string dtype) {
  const auto tgt_dtype = [&]() {
    if (dtype == "uint8")
      return torch::kUInt8;
    else if (dtype == "int16")
      return torch::kInt16;
    else if (dtype == "int32")
      return torch::kInt32;
    else if (dtype == "float32")
      return torch::kFloat32;
    else if (dtype == "float64")
      return torch::kFloat64;
    else
      throw std::runtime_error("Unsupported dtype");
  }();
  return c10::scalarTypeToTypeMeta(tgt_dtype);
}

torch::Tensor convert_to_tensor(
    sox_sample_t* buffer,
    const int32_t num_samples,
    const int32_t num_channels,
    const caffe2::TypeMeta dtype,
    const bool normalize,
    const bool channels_first) {
  torch::Tensor t;
  uint64_t dummy;
  SOX_SAMPLE_LOCALS;
  if (normalize || dtype == torch::kFloat32) {
    t = torch::empty(
        {num_samples / num_channels, num_channels}, torch::kFloat32);
    auto ptr = t.data_ptr<float_t>();
    for (int32_t i = 0; i < num_samples; ++i) {
      ptr[i] = SOX_SAMPLE_TO_FLOAT_32BIT(buffer[i], dummy);
    }
  } else if (dtype == torch::kInt32) {
    t = torch::from_blob(
            buffer, {num_samples / num_channels, num_channels}, torch::kInt32)
            .clone();
  } else if (dtype == torch::kInt16) {
    t = torch::empty({num_samples / num_channels, num_channels}, torch::kInt16);
    auto ptr = t.data_ptr<int16_t>();
    for (int32_t i = 0; i < num_samples; ++i) {
      ptr[i] = SOX_SAMPLE_TO_SIGNED_16BIT(buffer[i], dummy);
    }
  } else if (dtype == torch::kUInt8) {
    t = torch::empty({num_samples / num_channels, num_channels}, torch::kUInt8);
    auto ptr = t.data_ptr<uint8_t>();
    for (int32_t i = 0; i < num_samples; ++i) {
      ptr[i] = SOX_SAMPLE_TO_UNSIGNED_8BIT(buffer[i], dummy);
    }
  } else {
    throw std::runtime_error("Unsupported dtype.");
  }
  if (channels_first) {
    t = t.transpose(1, 0);
  }
  return t.contiguous();
}

const std::string get_filetype(const std::string path) {
  std::string ext = path.substr(path.find_last_of(".") + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext;
}

namespace {

std::tuple<sox_encoding_t, unsigned> get_save_encoding_for_wav(
    const std::string format,
    const caffe2::TypeMeta dtype,
    const Encoding& encoding,
    const BitDepth& bits_per_sample) {
  switch (encoding) {
    case Encoding::NOT_PROVIDED:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
          if (dtype == torch::kFloat32)
            return std::make_tuple<>(SOX_ENCODING_FLOAT, 32);
          if (dtype == torch::kInt32)
            return std::make_tuple<>(SOX_ENCODING_SIGN2, 32);
          if (dtype == torch::kInt16)
            return std::make_tuple<>(SOX_ENCODING_SIGN2, 16);
          if (dtype == torch::kUInt8)
            return std::make_tuple<>(SOX_ENCODING_UNSIGNED, 8);
          throw std::runtime_error("Internal Error: Unexpected dtype.");
        case BitDepth::B8:
          return std::make_tuple<>(SOX_ENCODING_UNSIGNED, 8);
        default:
          return std::make_tuple<>(
              SOX_ENCODING_SIGN2, static_cast<unsigned>(bits_per_sample));
      }
    case Encoding::PCM_SIGNED:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
          return std::make_tuple<>(SOX_ENCODING_SIGN2, 32);
        case BitDepth::B8:
          throw std::runtime_error(
              format + " does not support 8-bit signed PCM encoding.");
        default:
          return std::make_tuple<>(
              SOX_ENCODING_SIGN2, static_cast<unsigned>(bits_per_sample));
      }
    case Encoding::PCM_UNSIGNED:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
        case BitDepth::B8:
          return std::make_tuple<>(SOX_ENCODING_UNSIGNED, 8);
        default:
          throw std::runtime_error(
              format + " only supports 8-bit for unsigned PCM encoding.");
      }
    case Encoding::PCM_FLOAT:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
        case BitDepth::B32:
          return std::make_tuple<>(SOX_ENCODING_FLOAT, 32);
        case BitDepth::B64:
          return std::make_tuple<>(SOX_ENCODING_FLOAT, 64);
        default:
          throw std::runtime_error(
              format +
              " only supports 32-bit or 64-bit for floating-point PCM encoding.");
      }
    case Encoding::ULAW:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
        case BitDepth::B8:
          return std::make_tuple<>(SOX_ENCODING_ULAW, 8);
        default:
          throw std::runtime_error(
              format + " only supports 8-bit for mu-law encoding.");
      }
    case Encoding::ALAW:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
        case BitDepth::B8:
          return std::make_tuple<>(SOX_ENCODING_ALAW, 8);
        default:
          throw std::runtime_error(
              format + " only supports 8-bit for a-law encoding.");
      }
    default:
      throw std::runtime_error(
          format + " does not support encoding: " + to_string(encoding));
  }
}

std::tuple<sox_encoding_t, unsigned> get_save_encoding(
    const std::string& format,
    const caffe2::TypeMeta dtype,
    const c10::optional<std::string>& encoding,
    const c10::optional<int64_t>& bits_per_sample) {
  const Format fmt = get_format_from_string(format);
  const Encoding enc = get_encoding_from_option(encoding);
  const BitDepth bps = get_bit_depth_from_option(bits_per_sample);

  switch (fmt) {
    case Format::WAV:
    case Format::AMB:
      return get_save_encoding_for_wav(format, dtype, enc, bps);
    case Format::MP3:
      if (enc != Encoding::NOT_PROVIDED)
        throw std::runtime_error("mp3 does not support `encoding` option.");
      if (bps != BitDepth::NOT_PROVIDED)
        throw std::runtime_error(
            "mp3 does not support `bits_per_sample` option.");
      return std::make_tuple<>(SOX_ENCODING_MP3, 16);
    case Format::VORBIS:
      if (enc != Encoding::NOT_PROVIDED)
        throw std::runtime_error("vorbis does not support `encoding` option.");
      if (bps != BitDepth::NOT_PROVIDED)
        throw std::runtime_error(
            "vorbis does not support `bits_per_sample` option.");
      return std::make_tuple<>(SOX_ENCODING_VORBIS, 16);
    case Format::AMR_NB:
      if (enc != Encoding::NOT_PROVIDED)
        throw std::runtime_error("amr-nb does not support `encoding` option.");
      if (bps != BitDepth::NOT_PROVIDED)
        throw std::runtime_error(
            "amr-nb does not support `bits_per_sample` option.");
      return std::make_tuple<>(SOX_ENCODING_AMR_NB, 16);
    case Format::FLAC:
      if (enc != Encoding::NOT_PROVIDED)
        throw std::runtime_error("flac does not support `encoding` option.");
      switch (bps) {
        case BitDepth::B32:
        case BitDepth::B64:
          throw std::runtime_error(
              "flac does not support `bits_per_sample` larger than 24.");
        default:
          return std::make_tuple<>(
              SOX_ENCODING_FLAC, static_cast<unsigned>(bps));
      }
    case Format::SPHERE:
      switch (enc) {
        case Encoding::NOT_PROVIDED:
        case Encoding::PCM_SIGNED:
          switch (bps) {
            case BitDepth::NOT_PROVIDED:
              return std::make_tuple<>(SOX_ENCODING_SIGN2, 32);
            default:
              return std::make_tuple<>(
                  SOX_ENCODING_SIGN2, static_cast<unsigned>(bps));
          }
        case Encoding::PCM_UNSIGNED:
          throw std::runtime_error(
              "sph does not support unsigned integer PCM.");
        case Encoding::PCM_FLOAT:
          throw std::runtime_error("sph does not support floating point PCM.");
        case Encoding::ULAW:
          switch (bps) {
            case BitDepth::NOT_PROVIDED:
            case BitDepth::B8:
              return std::make_tuple<>(SOX_ENCODING_ULAW, 8);
            default:
              throw std::runtime_error(
                  "sph only supports 8-bit for mu-law encoding.");
          }
        case Encoding::ALAW:
          switch (bps) {
            case BitDepth::NOT_PROVIDED:
            case BitDepth::B8:
              return std::make_tuple<>(SOX_ENCODING_ALAW, 8);
            default:
              return std::make_tuple<>(
                  SOX_ENCODING_ALAW, static_cast<unsigned>(bps));
          }
        default:
          throw std::runtime_error(
              "sph does not support encoding: " + encoding.value());
      }
    default:
      throw std::runtime_error("Unsupported format: " + format);
  }
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
    return 16;
  }
  throw std::runtime_error("Unsupported file type: " + filetype);
}

} // namespace

sox_signalinfo_t get_signalinfo(
    const torch::Tensor* waveform,
    const int64_t sample_rate,
    const std::string filetype,
    const bool channels_first) {
  return sox_signalinfo_t{
      /*rate=*/static_cast<sox_rate_t>(sample_rate),
      /*channels=*/
      static_cast<unsigned>(waveform->size(channels_first ? 0 : 1)),
      /*precision=*/get_precision(filetype, waveform->dtype()),
      /*length=*/static_cast<uint64_t>(waveform->numel())};
}

sox_encodinginfo_t get_tensor_encodinginfo(const caffe2::TypeMeta dtype) {
  sox_encoding_t encoding = [&]() {
    if (dtype == torch::kUInt8)
      return SOX_ENCODING_UNSIGNED;
    if (dtype == torch::kInt16)
      return SOX_ENCODING_SIGN2;
    if (dtype == torch::kInt32)
      return SOX_ENCODING_SIGN2;
    if (dtype == torch::kFloat32)
      return SOX_ENCODING_FLOAT;
    throw std::runtime_error("Unsupported dtype.");
  }();
  unsigned bits_per_sample = [&]() {
    if (dtype == torch::kUInt8)
      return 8;
    if (dtype == torch::kInt16)
      return 16;
    if (dtype == torch::kInt32)
      return 32;
    if (dtype == torch::kFloat32)
      return 32;
    throw std::runtime_error("Unsupported dtype.");
  }();
  return sox_encodinginfo_t{
      /*encoding=*/encoding,
      /*bits_per_sample=*/bits_per_sample,
      /*compression=*/HUGE_VAL,
      /*reverse_bytes=*/sox_option_default,
      /*reverse_nibbles=*/sox_option_default,
      /*reverse_bits=*/sox_option_default,
      /*opposite_endian=*/sox_false};
}

sox_encodinginfo_t get_encodinginfo_for_save(
    const std::string& format,
    const caffe2::TypeMeta dtype,
    const c10::optional<double>& compression,
    const c10::optional<std::string>& encoding,
    const c10::optional<int64_t>& bits_per_sample) {
  auto enc = get_save_encoding(format, dtype, encoding, bits_per_sample);
  return sox_encodinginfo_t{
      /*encoding=*/std::get<0>(enc),
      /*bits_per_sample=*/std::get<1>(enc),
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

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::sox_utils_set_seed", &torchaudio::sox_utils::set_seed);
  m.def(
      "torchaudio::sox_utils_set_verbosity",
      &torchaudio::sox_utils::set_verbosity);
  m.def(
      "torchaudio::sox_utils_set_use_threads",
      &torchaudio::sox_utils::set_use_threads);
  m.def(
      "torchaudio::sox_utils_set_buffer_size",
      &torchaudio::sox_utils::set_buffer_size);
  m.def(
      "torchaudio::sox_utils_list_effects",
      &torchaudio::sox_utils::list_effects);
  m.def(
      "torchaudio::sox_utils_list_read_formats",
      &torchaudio::sox_utils::list_read_formats);
  m.def(
      "torchaudio::sox_utils_list_write_formats",
      &torchaudio::sox_utils::list_write_formats);
}

} // namespace sox_utils
} // namespace torchaudio
