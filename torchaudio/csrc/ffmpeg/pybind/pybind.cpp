#include <torch/extension.h>
#include <torchaudio/csrc/ffmpeg/hw_context.h>
#include <torchaudio/csrc/ffmpeg/pybind/fileobj.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/stream_reader.h>
#include <torchaudio/csrc/ffmpeg/stream_writer/stream_writer.h>

namespace torchaudio {
namespace io {
namespace {

std::map<std::string, std::tuple<int64_t, int64_t, int64_t>> get_versions() {
  std::map<std::string, std::tuple<int64_t, int64_t, int64_t>> ret;

#define add_version(NAME)            \
  {                                  \
    int ver = NAME##_version();      \
    ret.emplace(                     \
        "lib" #NAME,                 \
        std::make_tuple<>(           \
            AV_VERSION_MAJOR(ver),   \
            AV_VERSION_MINOR(ver),   \
            AV_VERSION_MICRO(ver))); \
  }

  add_version(avutil);
  add_version(avcodec);
  add_version(avformat);
  add_version(avfilter);
  add_version(avdevice);
  return ret;

#undef add_version
}

std::map<std::string, std::string> get_demuxers(bool req_device) {
  std::map<std::string, std::string> ret;
  const AVInputFormat* fmt = nullptr;
  void* i = nullptr;
  while ((fmt = av_demuxer_iterate(&i))) {
    assert(fmt);
    bool is_device = [&]() {
      const AVClass* avclass = fmt->priv_class;
      return avclass && AV_IS_INPUT_DEVICE(avclass->category);
    }();
    if (req_device == is_device) {
      ret.emplace(fmt->name, fmt->long_name);
    }
  }
  return ret;
}

std::map<std::string, std::string> get_muxers(bool req_device) {
  std::map<std::string, std::string> ret;
  const AVOutputFormat* fmt = nullptr;
  void* i = nullptr;
  while ((fmt = av_muxer_iterate(&i))) {
    assert(fmt);
    bool is_device = [&]() {
      const AVClass* avclass = fmt->priv_class;
      return avclass && AV_IS_OUTPUT_DEVICE(avclass->category);
    }();
    if (req_device == is_device) {
      ret.emplace(fmt->name, fmt->long_name);
    }
  }
  return ret;
}

std::map<std::string, std::string> get_codecs(
    AVMediaType type,
    bool req_encoder) {
  const AVCodec* c = nullptr;
  void* i = nullptr;
  std::map<std::string, std::string> ret;
  while ((c = av_codec_iterate(&i))) {
    assert(c);
    if ((req_encoder && av_codec_is_encoder(c)) ||
        (!req_encoder && av_codec_is_decoder(c))) {
      if (c->type == type && c->name) {
        ret.emplace(c->name, c->long_name ? c->long_name : "");
      }
    }
  }
  return ret;
}

std::vector<std::string> get_protocols(bool output) {
  void* opaque = nullptr;
  const char* name = nullptr;
  std::vector<std::string> ret;
  while ((name = avio_enum_protocols(&opaque, output))) {
    assert(name);
    ret.emplace_back(name);
  }
  return ret;
}

std::string get_build_config() {
  return avcodec_configuration();
}

// The reason we inherit FileObj instead of making it an attribute
// is so that FileObj is instantiated first.
// AVIOContext must be initialized before AVFormat, and outlive AVFormat.
struct StreamReaderFileObj : private FileObj, public StreamReader {
  StreamReaderFileObj(
      py::object fileobj,
      const c10::optional<std::string>& format,
      const c10::optional<std::map<std::string, std::string>>& option,
      int64_t buffer_size)
      : FileObj(fileobj, static_cast<int>(buffer_size), false),
        StreamReader(pAVIO, format, option) {}
};

struct StreamWriterFileObj : private FileObj, public StreamWriter {
  StreamWriterFileObj(
      py::object fileobj,
      const c10::optional<std::string>& format,
      int64_t buffer_size)
      : FileObj(fileobj, static_cast<int>(buffer_size), true),
        StreamWriter(pAVIO, format) {}
};

PYBIND11_MODULE(_torchaudio_ffmpeg, m) {
  m.def("init", []() { avdevice_register_all(); });
  m.def("get_log_level", []() { return av_log_get_level(); });
  m.def("set_log_level", [](int level) { av_log_set_level(level); });
  m.def("get_versions", &get_versions);
  m.def("get_muxers", []() { return get_muxers(false); });
  m.def("get_demuxers", []() { return get_demuxers(false); });
  m.def("get_input_devices", []() { return get_demuxers(true); });
  m.def("get_build_config", &get_build_config);
  m.def("get_output_devices", []() { return get_muxers(true); });
  m.def("get_audio_decoders", []() {
    return get_codecs(AVMEDIA_TYPE_AUDIO, false);
  });
  m.def("get_audio_encoders", []() {
    return get_codecs(AVMEDIA_TYPE_AUDIO, true);
  });
  m.def("get_video_decoders", []() {
    return get_codecs(AVMEDIA_TYPE_VIDEO, false);
  });
  m.def("get_video_encoders", []() {
    return get_codecs(AVMEDIA_TYPE_VIDEO, true);
  });
  m.def("get_input_protocols", []() { return get_protocols(false); });
  m.def("get_output_protocols", []() { return get_protocols(true); });
  m.def("clear_cuda_context_cache", &clear_cuda_context_cache);

  py::class_<Chunk>(m, "Chunk", py::module_local())
      .def_readwrite("frames", &Chunk::frames)
      .def_readwrite("pts", &Chunk::pts);
  py::class_<CodecConfig>(m, "CodecConfig", py::module_local())
      .def(py::init<int, int, const c10::optional<int>&, int, int>());
  py::class_<StreamWriter>(m, "StreamWriter", py::module_local())
      .def(py::init<const std::string&, const c10::optional<std::string>&>())
      .def("set_metadata", &StreamWriter::set_metadata)
      .def("add_audio_stream", &StreamWriter::add_audio_stream)
      .def("add_video_stream", &StreamWriter::add_video_stream)
      .def("dump_format", &StreamWriter::dump_format)
      .def("open", &StreamWriter::open)
      .def("write_audio_chunk", &StreamWriter::write_audio_chunk)
      .def("write_video_chunk", &StreamWriter::write_video_chunk)
      .def("flush", &StreamWriter::flush)
      .def("close", &StreamWriter::close);
  py::class_<StreamWriterFileObj>(m, "StreamWriterFileObj", py::module_local())
      .def(py::init<py::object, const c10::optional<std::string>&, int64_t>())
      .def("set_metadata", &StreamWriterFileObj::set_metadata)
      .def("add_audio_stream", &StreamWriterFileObj::add_audio_stream)
      .def("add_video_stream", &StreamWriterFileObj::add_video_stream)
      .def("dump_format", &StreamWriterFileObj::dump_format)
      .def("open", &StreamWriterFileObj::open)
      .def("write_audio_chunk", &StreamWriterFileObj::write_audio_chunk)
      .def("write_video_chunk", &StreamWriterFileObj::write_video_chunk)
      .def("flush", &StreamWriterFileObj::flush)
      .def("close", &StreamWriterFileObj::close);
  py::class_<OutputStreamInfo>(m, "OutputStreamInfo", py::module_local())
      .def_readonly("source_index", &OutputStreamInfo::source_index)
      .def_readonly("filter_description", &OutputStreamInfo::filter_description)
      .def_property_readonly(
          "media_type",
          [](const OutputStreamInfo& o) -> std::string {
            return av_get_media_type_string(o.media_type);
          })
      .def_property_readonly(
          "format",
          [](const OutputStreamInfo& o) -> std::string {
            switch (o.media_type) {
              case AVMEDIA_TYPE_AUDIO:
                return av_get_sample_fmt_name((AVSampleFormat)(o.format));
              case AVMEDIA_TYPE_VIDEO:
                return av_get_pix_fmt_name((AVPixelFormat)(o.format));
              default:
                TORCH_INTERNAL_ASSERT(
                    false,
                    "FilterGraph is returning unexpected media type: ",
                    av_get_media_type_string(o.media_type));
            }
          })
      .def_readonly("sample_rate", &OutputStreamInfo::sample_rate)
      .def_readonly("num_channels", &OutputStreamInfo::num_channels)
      .def_readonly("width", &OutputStreamInfo::width)
      .def_readonly("height", &OutputStreamInfo::height)
      .def_property_readonly(
          "frame_rate", [](const OutputStreamInfo& o) -> double {
            if (o.frame_rate.den == 0) {
              TORCH_WARN(
                  o.frame_rate.den,
                  "Invalid frame rate is found: ",
                  o.frame_rate.num,
                  "/",
                  o.frame_rate.den);
              return -1;
            }
            return static_cast<double>(o.frame_rate.num) / o.frame_rate.den;
          });
  py::class_<SrcStreamInfo>(m, "SourceStreamInfo", py::module_local())
      .def_property_readonly(
          "media_type",
          [](const SrcStreamInfo& s) {
            return av_get_media_type_string(s.media_type);
          })
      .def_readonly("codec_name", &SrcStreamInfo::codec_name)
      .def_readonly("codec_long_name", &SrcStreamInfo::codec_long_name)
      .def_readonly("format", &SrcStreamInfo::fmt_name)
      .def_readonly("bit_rate", &SrcStreamInfo::bit_rate)
      .def_readonly("num_frames", &SrcStreamInfo::num_frames)
      .def_readonly("bits_per_sample", &SrcStreamInfo::bits_per_sample)
      .def_readonly("metadata", &SrcStreamInfo::metadata)
      .def_readonly("sample_rate", &SrcStreamInfo::sample_rate)
      .def_readonly("num_channels", &SrcStreamInfo::num_channels)
      .def_readonly("width", &SrcStreamInfo::width)
      .def_readonly("height", &SrcStreamInfo::height)
      .def_readonly("frame_rate", &SrcStreamInfo::frame_rate);
  py::class_<StreamReader>(m, "StreamReader", py::module_local())
      .def(py::init<
           const std::string&,
           const c10::optional<std::string>&,
           const c10::optional<OptionDict>&>())
      .def("num_src_streams", &StreamReader::num_src_streams)
      .def("num_out_streams", &StreamReader::num_out_streams)
      .def("find_best_audio_stream", &StreamReader::find_best_audio_stream)
      .def("find_best_video_stream", &StreamReader::find_best_video_stream)
      .def("get_metadata", &StreamReader::get_metadata)
      .def("get_src_stream_info", &StreamReader::get_src_stream_info)
      .def("get_out_stream_info", &StreamReader::get_out_stream_info)
      .def("seek", &StreamReader::seek)
      .def("add_audio_stream", &StreamReader::add_audio_stream)
      .def("add_video_stream", &StreamReader::add_video_stream)
      .def("remove_stream", &StreamReader::remove_stream)
      .def(
          "process_packet",
          py::overload_cast<const c10::optional<double>&, const double>(
              &StreamReader::process_packet))
      .def("process_all_packets", &StreamReader::process_all_packets)
      .def("fill_buffer", &StreamReader::fill_buffer)
      .def("is_buffer_ready", &StreamReader::is_buffer_ready)
      .def("pop_chunks", &StreamReader::pop_chunks);
  py::class_<StreamReaderFileObj>(m, "StreamReaderFileObj", py::module_local())
      .def(py::init<
           py::object,
           const c10::optional<std::string>&,
           const c10::optional<OptionDict>&,
           int64_t>())
      .def("num_src_streams", &StreamReaderFileObj::num_src_streams)
      .def("num_out_streams", &StreamReaderFileObj::num_out_streams)
      .def(
          "find_best_audio_stream",
          &StreamReaderFileObj::find_best_audio_stream)
      .def(
          "find_best_video_stream",
          &StreamReaderFileObj::find_best_video_stream)
      .def("get_metadata", &StreamReaderFileObj::get_metadata)
      .def("get_src_stream_info", &StreamReaderFileObj::get_src_stream_info)
      .def("get_out_stream_info", &StreamReaderFileObj::get_out_stream_info)
      .def("seek", &StreamReaderFileObj::seek)
      .def("add_audio_stream", &StreamReaderFileObj::add_audio_stream)
      .def("add_video_stream", &StreamReaderFileObj::add_video_stream)
      .def("remove_stream", &StreamReaderFileObj::remove_stream)
      .def(
          "process_packet",
          py::overload_cast<const c10::optional<double>&, const double>(
              &StreamReader::process_packet))
      .def("process_all_packets", &StreamReaderFileObj::process_all_packets)
      .def("fill_buffer", &StreamReaderFileObj::fill_buffer)
      .def("is_buffer_ready", &StreamReaderFileObj::is_buffer_ready)
      .def("pop_chunks", &StreamReaderFileObj::pop_chunks);
}

} // namespace
} // namespace io
} // namespace torchaudio
