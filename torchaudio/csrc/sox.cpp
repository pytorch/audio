#include <torchaudio/csrc/sox.h>
#include <torchaudio/csrc/sox_effects.h>
#include <torchaudio/csrc/sox_io.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace torch {
namespace audio {
namespace {
/// Helper struct to safely close the sox_format_t descriptor.
struct SoxDescriptor {
  explicit SoxDescriptor(sox_format_t* fd) noexcept : fd_(fd) {}
  SoxDescriptor(const SoxDescriptor& other) = delete;
  SoxDescriptor(SoxDescriptor&& other) = delete;
  SoxDescriptor& operator=(const SoxDescriptor& other) = delete;
  SoxDescriptor& operator=(SoxDescriptor&& other) = delete;
  ~SoxDescriptor() {
    if (fd_ != nullptr) {
      sox_close(fd_);
    }
  }
  sox_format_t* operator->() noexcept {
    return fd_;
  }
  sox_format_t* get() noexcept {
    return fd_;
  }

 private:
  sox_format_t* fd_;
};

int64_t write_audio(SoxDescriptor& fd, at::Tensor tensor) {
  std::vector<sox_sample_t> buffer(tensor.numel());

  AT_DISPATCH_ALL_TYPES(tensor.scalar_type(), "write_audio_buffer", [&] {
    auto* data = tensor.data_ptr<scalar_t>();
    std::copy(data, data + tensor.numel(), buffer.begin());
  });

  const auto samples_written =
      sox_write(fd.get(), buffer.data(), buffer.size());

  return samples_written;
}

void read_audio(
    SoxDescriptor& fd,
    at::Tensor output,
    int64_t buffer_length) {
  std::vector<sox_sample_t> buffer(buffer_length);

  int number_of_channels = fd->signal.channels;
  const int64_t samples_read = sox_read(fd.get(), buffer.data(), buffer_length);
  if (samples_read == 0) {
    throw std::runtime_error(
        "Error reading audio file: empty file or read failed in sox_read");
  }

  output.resize_({samples_read / number_of_channels, number_of_channels});
  output = output.contiguous();

  AT_DISPATCH_ALL_TYPES(output.scalar_type(), "read_audio_buffer", [&] {
    auto* data = output.data_ptr<scalar_t>();
    std::copy(buffer.begin(), buffer.begin() + samples_read, data);
  });
}
} // namespace

std::tuple<sox_signalinfo_t, sox_encodinginfo_t> get_info(
    const std::string& file_name
  ) {
  SoxDescriptor fd(sox_open_read(
      file_name.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));
  if (fd.get() == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }
  return std::make_tuple(fd->signal, fd->encoding);
}

int read_audio_file(
    const std::string& file_name,
    at::Tensor output,
    bool ch_first,
    int64_t nframes,
    int64_t offset,
    sox_signalinfo_t* si,
    sox_encodinginfo_t* ei,
    const char* ft) {

  SoxDescriptor fd(sox_open_read(file_name.c_str(), si, ei, ft));
  if (fd.get() == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }

  // signal info

  const int number_of_channels = fd->signal.channels;
  const int sample_rate = fd->signal.rate;
  const int64_t total_length = fd->signal.length;

  // multiply offset and number of frames by number of channels
  offset *= number_of_channels;
  nframes *= number_of_channels;

  if (total_length == 0) {
    throw std::runtime_error("Error reading audio file: unknown length");
  }
  if (offset > total_length) {
    throw std::runtime_error("Offset past EOF");
  }

  // calculate buffer length
  int64_t buffer_length = total_length;
  if (offset > 0) {
      buffer_length -= offset;
  }
  if (nframes > 0 && buffer_length > nframes) {
      buffer_length = nframes;
  }

  // seek to offset point before reading data
  if (sox_seek(fd.get(), offset, 0) == SOX_EOF) {
    throw std::runtime_error("sox_seek reached EOF, try reducing offset or num_samples");
  }

  // read data and fill output tensor
  read_audio(fd, output, buffer_length);

  // L x C -> C x L, if desired
  if (ch_first) {
    output.transpose_(1, 0);
  }

  return sample_rate;
}

void write_audio_file(
    const std::string& file_name,
    const at::Tensor& tensor,
    sox_signalinfo_t* si,
    sox_encodinginfo_t* ei,
    const char* file_type) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(
        "Error writing audio file: input tensor must be contiguous");
  }

#if SOX_LIB_VERSION_CODE >= 918272 // >= 14.3.0
  si->mult = nullptr;
#endif

  SoxDescriptor fd(sox_open_write(
      file_name.c_str(),
      si,
      ei,
      file_type,
      /*oob=*/nullptr,
      /*overwrite=*/nullptr));

  if (fd.get() == nullptr) {
    throw std::runtime_error(
        "Error writing audio file: could not open file for writing");
  }

  const auto samples_written = write_audio(fd, tensor);

  if (samples_written != tensor.numel()) {
    throw std::runtime_error(
        "Error writing audio file: could not write entire buffer");
  }
}

} // namespace audio
} // namespace torch

namespace {

using SignalInfo = std::tuple<int64_t, int64_t, int64_t>;
SignalInfo get_info_bytes(
    py::bytes src,
    c10::optional<std::string>& format) {
  auto bytes = static_cast<std::string>(src);
  auto info = torchaudio::sox_io::get_info_bytes(bytes, format);
  return std::make_tuple(
      info->getSampleRate(), info->getNumFrames(), info->getNumChannels());
}

std::tuple<torch::Tensor, int64_t> load_audio_bytes(
    py::bytes src,
    c10::optional<int64_t> frame_offset,
    c10::optional<int64_t> num_frames,
    c10::optional<bool> normalize,
    c10::optional<bool> channels_first,
    c10::optional<std::string>& format) {
  auto bytes = static_cast<std::string>(src);
  auto result = torchaudio::sox_io::load_audio_bytes(
      bytes, frame_offset, num_frames, normalize, channels_first, format);
  return std::make_tuple(result->getTensor(), result->getSampleRate());
}

std::tuple<torch::Tensor, int64_t> apply_effects_bytes(
    py::bytes src,
    std::vector<std::vector<std::string>> effects,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
    c10::optional<std::string>& format) {
  auto bytes = static_cast<std::string>(src);
  auto result = torchaudio::sox_effects::apply_effects_bytes(
      bytes, effects, normalize, channels_first, format);
  return std::make_tuple(result->getTensor(), result->getSampleRate());
}

} // namespace

PYBIND11_MODULE(_torchaudio, m) {
  py::class_<sox_signalinfo_t>(m, "sox_signalinfo_t")
       .def(py::init<>())
       .def("__repr__", [](const sox_signalinfo_t &self) {
         std::stringstream ss;
         ss << "sox_signalinfo_t {\n"
            << "  rate-> " << self.rate << "\n"
            << "  channels-> " << self.channels << "\n"
            << "  precision-> " << self.precision << "\n"
            << "  length-> " << self.length << "\n"
            << "  mult-> " << self.mult << "\n"
            << "}\n";
         return ss.str();
       })
       .def_readwrite("rate", &sox_signalinfo_t::rate)
       .def_readwrite("channels", &sox_signalinfo_t::channels)
       .def_readwrite("precision", &sox_signalinfo_t::precision)
       .def_readwrite("length", &sox_signalinfo_t::length)
       .def_readwrite("mult", &sox_signalinfo_t::mult);
  py::class_<sox_encodinginfo_t>(m, "sox_encodinginfo_t")
       .def(py::init<>())
       .def("__repr__", [](const sox_encodinginfo_t &self) {
         std::stringstream ss;
         ss << "sox_encodinginfo_t {\n"
            << "  encoding-> " << self.encoding << "\n"
            << "  bits_per_sample-> " << self.bits_per_sample << "\n"
            << "  compression-> " << self.compression << "\n"
            << "  reverse_bytes-> " << self.reverse_bytes << "\n"
            << "  reverse_nibbles-> " << self.reverse_nibbles << "\n"
            << "  reverse_bits-> " << self.reverse_bits << "\n"
            << "  opposite_endian-> " << self.opposite_endian << "\n"
            << "}\n";
         return ss.str();
       })
       .def_readwrite("encoding", &sox_encodinginfo_t::encoding)
       .def_readwrite("bits_per_sample", &sox_encodinginfo_t::bits_per_sample)
       .def_readwrite("compression", &sox_encodinginfo_t::compression)
       .def_readwrite("reverse_bytes", &sox_encodinginfo_t::reverse_bytes)
       .def_readwrite("reverse_nibbles", &sox_encodinginfo_t::reverse_nibbles)
       .def_readwrite("reverse_bits", &sox_encodinginfo_t::reverse_bits)
       .def_readwrite("opposite_endian", &sox_encodinginfo_t::opposite_endian);
  py::enum_<sox_encoding_t>(m, "sox_encoding_t")
       .value("SOX_ENCODING_UNKNOWN", sox_encoding_t::SOX_ENCODING_UNKNOWN)
       .value("SOX_ENCODING_SIGN2", sox_encoding_t::SOX_ENCODING_SIGN2)
       .value("SOX_ENCODING_UNSIGNED", sox_encoding_t::SOX_ENCODING_UNSIGNED)
       .value("SOX_ENCODING_FLOAT", sox_encoding_t::SOX_ENCODING_FLOAT)
       .value("SOX_ENCODING_FLOAT_TEXT", sox_encoding_t::SOX_ENCODING_FLOAT_TEXT)
       .value("SOX_ENCODING_FLAC", sox_encoding_t::SOX_ENCODING_FLAC)
       .value("SOX_ENCODING_HCOM", sox_encoding_t::SOX_ENCODING_HCOM)
       .value("SOX_ENCODING_WAVPACK", sox_encoding_t::SOX_ENCODING_WAVPACK)
       .value("SOX_ENCODING_WAVPACKF", sox_encoding_t::SOX_ENCODING_WAVPACKF)
       .value("SOX_ENCODING_ULAW", sox_encoding_t::SOX_ENCODING_ULAW)
       .value("SOX_ENCODING_ALAW", sox_encoding_t::SOX_ENCODING_ALAW)
       .value("SOX_ENCODING_G721", sox_encoding_t::SOX_ENCODING_G721)
       .value("SOX_ENCODING_G723", sox_encoding_t::SOX_ENCODING_G723)
       .value("SOX_ENCODING_CL_ADPCM", sox_encoding_t::SOX_ENCODING_CL_ADPCM)
       .value("SOX_ENCODING_CL_ADPCM16", sox_encoding_t::SOX_ENCODING_CL_ADPCM16)
       .value("SOX_ENCODING_MS_ADPCM", sox_encoding_t::SOX_ENCODING_MS_ADPCM)
       .value("SOX_ENCODING_IMA_ADPCM", sox_encoding_t::SOX_ENCODING_IMA_ADPCM)
       .value("SOX_ENCODING_OKI_ADPCM", sox_encoding_t::SOX_ENCODING_OKI_ADPCM)
       .value("SOX_ENCODING_DPCM", sox_encoding_t::SOX_ENCODING_DPCM)
       .value("SOX_ENCODING_DWVW", sox_encoding_t::SOX_ENCODING_DWVW)
       .value("SOX_ENCODING_DWVWN", sox_encoding_t::SOX_ENCODING_DWVWN)
       .value("SOX_ENCODING_GSM", sox_encoding_t::SOX_ENCODING_GSM)
       .value("SOX_ENCODING_MP3", sox_encoding_t::SOX_ENCODING_MP3)
       .value("SOX_ENCODING_VORBIS", sox_encoding_t::SOX_ENCODING_VORBIS)
       .value("SOX_ENCODING_AMR_WB", sox_encoding_t::SOX_ENCODING_AMR_WB)
       .value("SOX_ENCODING_AMR_NB", sox_encoding_t::SOX_ENCODING_AMR_NB)
       .value("SOX_ENCODING_LPC10", sox_encoding_t::SOX_ENCODING_LPC10)
       //.value("SOX_ENCODING_OPUS", sox_encoding_t::SOX_ENCODING_OPUS)  // creates a compile error
       .value("SOX_ENCODINGS", sox_encoding_t::SOX_ENCODINGS)
       .export_values();
  py::enum_<sox_option_t>(m, "sox_option_t")
       .value("sox_option_no", sox_option_t::sox_option_no)
       .value("sox_option_yes", sox_option_t::sox_option_yes)
       .value("sox_option_default", sox_option_t::sox_option_default)
       .export_values();
  py::enum_<sox_bool>(m, "sox_bool")
       .value("sox_false", sox_bool::sox_false)
       .value("sox_true", sox_bool::sox_true)
       .export_values();
  m.def(
      "read_audio_file",
      &torch::audio::read_audio_file,
      "Reads an audio file into a tensor");
  m.def(
      "write_audio_file",
      &torch::audio::write_audio_file,
      "Writes data from a tensor into an audio file");
  m.def(
      "get_info",
      &torch::audio::get_info,
      "Gets information about an audio file");
  m.def(
      "get_info_bytes", &get_info_bytes, "Get information about an audio file");
  m.def("load_audio_bytes", &load_audio_bytes, "Load audio from byte string.");
  m.def(
      "apply_effects_bytes",
      &apply_effects_bytes,
      "Decode audio data from byte string and apply sox effects.");
}
