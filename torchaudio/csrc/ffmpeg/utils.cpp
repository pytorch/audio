#include <torch/script.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace io {
namespace {

c10::Dict<std::string, std::tuple<int64_t, int64_t, int64_t>> get_versions() {
  c10::Dict<std::string, std::tuple<int64_t, int64_t, int64_t>> ret;

#define add_version(NAME)            \
  {                                  \
    int ver = NAME##_version();      \
    ret.insert(                      \
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

c10::Dict<std::string, std::string> get_demuxers(bool req_device) {
  c10::Dict<std::string, std::string> ret;
  const AVInputFormat* fmt = nullptr;
  void* i = nullptr;
  while ((fmt = av_demuxer_iterate(&i))) {
    assert(fmt);
    bool is_device = [&]() {
      const AVClass* avclass = fmt->priv_class;
      return avclass && AV_IS_INPUT_DEVICE(avclass->category);
    }();
    if (req_device == is_device) {
      ret.insert(fmt->name, fmt->long_name);
    }
  }
  return ret;
}

c10::Dict<std::string, std::string> get_muxers(bool req_device) {
  c10::Dict<std::string, std::string> ret;
  const AVOutputFormat* fmt = nullptr;
  void* i = nullptr;
  while ((fmt = av_muxer_iterate(&i))) {
    assert(fmt);
    bool is_device = [&]() {
      const AVClass* avclass = fmt->priv_class;
      return avclass && AV_IS_OUTPUT_DEVICE(avclass->category);
    }();
    if (req_device == is_device) {
      ret.insert(fmt->name, fmt->long_name);
    }
  }
  return ret;
}

c10::Dict<std::string, std::string> get_codecs(
    AVMediaType type,
    bool req_encoder) {
  const AVCodec* c = nullptr;
  void* i = nullptr;
  c10::Dict<std::string, std::string> ret;
  while ((c = av_codec_iterate(&i))) {
    assert(c);
    if ((req_encoder && av_codec_is_encoder(c)) ||
        (!req_encoder && av_codec_is_decoder(c))) {
      if (c->type == type && c->name) {
        ret.insert(c->name, c->long_name ? c->long_name : "");
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

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::ffmpeg_get_versions", &get_versions);
  m.def("torchaudio::ffmpeg_get_muxers", []() { return get_muxers(false); });
  m.def(
      "torchaudio::ffmpeg_get_demuxers", []() { return get_demuxers(false); });
  m.def("torchaudio::ffmpeg_get_input_devices", []() {
    return get_demuxers(true);
  });
  m.def("torchaudio::ffmpeg_get_output_devices", []() {
    return get_muxers(true);
  });
  m.def("torchaudio::ffmpeg_get_audio_decoders", []() {
    return get_codecs(AVMEDIA_TYPE_AUDIO, false);
  });
  m.def("torchaudio::ffmpeg_get_audio_encoders", []() {
    return get_codecs(AVMEDIA_TYPE_AUDIO, true);
  });
  m.def("torchaudio::ffmpeg_get_video_decoders", []() {
    return get_codecs(AVMEDIA_TYPE_VIDEO, false);
  });
  m.def("torchaudio::ffmpeg_get_video_encoders", []() {
    return get_codecs(AVMEDIA_TYPE_VIDEO, true);
  });
  m.def("torchaudio::ffmpeg_get_input_protocols", []() {
    return get_protocols(false);
  });
  m.def("torchaudio::ffmpeg_get_output_protocols", []() {
    return get_protocols(true);
  });
}

} // namespace
} // namespace io
} // namespace torchaudio
