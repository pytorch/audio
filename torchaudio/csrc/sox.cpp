#include <torchaudio/csrc/sox.h>

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

std::vector<std::string> get_effect_names() {
  sox_effect_fn_t const * fns = sox_get_effect_fns();
  std::vector<std::string> sv;
  for(int i = 0; fns[i]; ++i) {
    const sox_effect_handler_t *eh = fns[i] ();
    if(eh && eh->name)
      sv.push_back(eh->name);
  }
  return sv;
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

int initialize_sox() {
  /* Initialization for sox effects.  Only initialize once  */
  return sox_init();
}

int shutdown_sox() {
  /* Shutdown for sox effects.  Do not shutdown between multiple calls  */
  return sox_quit();
}

int build_flow_effects(const std::string& file_name,
                       at::Tensor otensor,
                       bool ch_first,
                       sox_signalinfo_t* target_signal,
                       sox_encodinginfo_t* target_encoding,
                       const char* file_type,
                       std::vector<SoxEffect> pyeffs,
                       int max_num_eopts) {

  /* This function builds an effects flow and puts the results into a tensor.
     It can also be used to re-encode audio using any of the available encoding
     options in SoX including sample rate and channel re-encoding.              */

  // open input
  sox_format_t* input = sox_open_read(file_name.c_str(), nullptr, nullptr, nullptr);
  if (input == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }

  // only used if target signal or encoding are null
  sox_signalinfo_t empty_signal;
  sox_encodinginfo_t empty_encoding;

  // set signalinfo and encodinginfo if blank
  if(target_signal == nullptr) {
    target_signal = &empty_signal;
    target_signal->rate = input->signal.rate;
    target_signal->channels = input->signal.channels;
    target_signal->length = SOX_UNSPEC;
    target_signal->precision = input->signal.precision;
#if SOX_LIB_VERSION_CODE >= 918272 // >= 14.3.0
    target_signal->mult = nullptr;
#endif
  }
  if(target_encoding == nullptr) {
    target_encoding = &empty_encoding;
    target_encoding->encoding = SOX_ENCODING_SIGN2; // Sample format
    target_encoding->bits_per_sample = input->signal.precision; // Bits per sample
    target_encoding->compression = 0.0; // Compression factor
    target_encoding->reverse_bytes = sox_option_default; // Should bytes be reversed
    target_encoding->reverse_nibbles = sox_option_default; // Should nibbles be reversed
    target_encoding->reverse_bits = sox_option_default; // Should bits be reversed (pairs of bits?)
    target_encoding->opposite_endian = sox_false; // Reverse endianness
  }

  // check for rate or channels effect and change the output signalinfo accordingly
  for (SoxEffect se : pyeffs) {
    if (se.ename == "rate") {
      target_signal->rate = std::stod(se.eopts[0]);
    } else if (se.ename == "channels") {
      target_signal->channels = std::stoi(se.eopts[0]);
    }
  }

  // create interm_signal for effects, intermediate steps change this in-place
  sox_signalinfo_t interm_signal = input->signal;

#ifdef __APPLE__
  // According to Mozilla Deepspeech sox_open_memstream_write doesn't work
  // with OSX
  char tmp_name[] = "/tmp/fileXXXXXX";
  int tmp_fd = mkstemp(tmp_name);
  close(tmp_fd);
  sox_format_t* output = sox_open_write(tmp_name, target_signal,
                                        target_encoding, "wav", nullptr, nullptr);
#else
  // create buffer and buffer_size for output in memwrite
  char* buffer;
  size_t buffer_size;
  // in-memory descriptor (this may not work for OSX)
  sox_format_t* output = sox_open_memstream_write(&buffer,
                                                  &buffer_size,
                                                  target_signal,
                                                  target_encoding,
                                                  file_type, nullptr);
#endif
  if (output == nullptr) {
    throw std::runtime_error("Error opening output memstream/temporary file");
  }
  // Setup the effects chain to decode/resample
  sox_effects_chain_t* chain =
    sox_create_effects_chain(&input->encoding, &output->encoding);

  sox_effect_t* e = sox_create_effect(sox_find_effect("input"));
  char* io_args[1];
  io_args[0] = (char*)input;
  sox_effect_options(e, 1, io_args);
  sox_add_effect(chain, e, &interm_signal, &input->signal);
  free(e);

  for(SoxEffect tae : pyeffs) {
    if(tae.ename == "no_effects") break;
    e = sox_create_effect(sox_find_effect(tae.ename.c_str()));
    e->global_info->global_info->verbosity = 1;
    if(tae.eopts[0] == "") {
      sox_effect_options(e, 0, nullptr);
    } else {
      int num_opts = tae.eopts.size();
      char* sox_args[max_num_eopts];
      for(std::vector<std::string>::size_type i = 0; i != tae.eopts.size(); i++) {
        sox_args[i] = (char*) tae.eopts[i].c_str();
      }
      if(sox_effect_options(e, num_opts, sox_args) != SOX_SUCCESS) {
#ifdef __APPLE__
        unlink(tmp_name);
#endif
        throw std::runtime_error("invalid effect options, see SoX docs for details");
      }
    }
    sox_add_effect(chain, e, &interm_signal, &output->signal);
    free(e);
  }

  e = sox_create_effect(sox_find_effect("output"));
  io_args[0] = (char*)output;
  sox_effect_options(e, 1, io_args);
  sox_add_effect(chain, e, &interm_signal, &output->signal);
  free(e);

  // Finally run the effects chain
  sox_flow_effects(chain, nullptr, nullptr);
  sox_delete_effects_chain(chain);

  // Close sox handles, buffer does not get properly sized until these are closed
  sox_close(output);
  sox_close(input);

  int sr;
  // Read the in-memory audio buffer or temp file that we just wrote.
#ifdef __APPLE__
  /*
     Temporary filetype must have a valid header.  Wav seems to work here while
     raw does not.  Certain effects like chorus caused strange behavior on the mac.
  */
  // read_audio_file reads the temporary file and returns the sr and otensor
  sr = read_audio_file(tmp_name, otensor, ch_first, 0, 0,
                       target_signal, target_encoding, "wav");
  // delete temporary audio file
  unlink(tmp_name);
#else
  // Resize output tensor to desired dimensions, different effects result in output->signal.length,
  // interm_signal.length and buffer size being inconsistent with the result of the file output.
  // We prioritize in the order: output->signal.length > interm_signal.length > buffer_size
  // Could be related to: https://sourceforge.net/p/sox/bugs/314/
  int nc, ns;
  if (output->signal.length == 0) {
    // sometimes interm_signal length is extremely large, but the buffer_size
    // is double the length of the output signal
    if (interm_signal.length > (buffer_size * 10)) {
      ns = buffer_size / 2;
    } else {
      ns = interm_signal.length;
    }
    nc = interm_signal.channels;
  } else {
    nc = output->signal.channels;
    ns = output->signal.length;
  }
  otensor.resize_({ns/nc, nc});
  otensor = otensor.contiguous();

  input = sox_open_mem_read(buffer, buffer_size, target_signal, target_encoding, file_type);
  std::vector<sox_sample_t> samples(buffer_size);
  const int64_t samples_read = sox_read(input, samples.data(), buffer_size);
  assert(samples_read != nc * ns && samples_read != 0);
  AT_DISPATCH_ALL_TYPES(otensor.scalar_type(), "effects_buffer", [&] {
    auto* data = otensor.data_ptr<scalar_t>();
    std::copy(samples.begin(), samples.begin() + samples_read, data);
  });
  // free buffer and close mem_read
  sox_close(input);
  free(buffer);

  if (ch_first) {
    otensor.transpose_(1, 0);
  }
  sr = target_signal->rate;

#endif
  // return sample rate, output tensor modified in-place
  return sr;
}
} // namespace audio
} // namespace torch

PYBIND11_MODULE(_torchaudio, m) {
  py::class_<torch::audio::SoxEffect>(m, "SoxEffect")
       .def(py::init<>())
       .def("__repr__", [](const torch::audio::SoxEffect &self) {
         std::stringstream ss;
         std::string sep;
         ss << "SoxEffect (" << self.ename << " ,[";
         for(std::string s : self.eopts) {
           ss << sep << "\"" << s << "\"";
           sep = ", ";
         }
         ss << "])\n";
         return ss.str();
       })
       .def_readwrite("ename", &torch::audio::SoxEffect::ename)
       .def_readwrite("eopts", &torch::audio::SoxEffect::eopts);
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
      "get_effect_names",
      &torch::audio::get_effect_names,
      "Gets the names of all available effects");
  m.def(
      "build_flow_effects",
      &torch::audio::build_flow_effects,
      "build effects and flow chain into tensors");
  m.def(
      "initialize_sox",
      &torch::audio::initialize_sox,
      "initialize sox for effects");
  m.def(
      "shutdown_sox",
      &torch::audio::shutdown_sox,
      "shutdown sox for effects");
}
