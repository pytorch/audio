#include <torch/extension.h>
#include <torchaudio/csrc/sox/io.h>
#include <torchaudio/csrc/sox/legacy.h>


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
      "load_audio_fileobj",
      &torchaudio::sox_io::load_audio_fileobj,
      "Load audio from file object.");
}
