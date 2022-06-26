#include <torch/script.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace ffmpeg {
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

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::ffmpeg_get_versions", &get_versions);
}

} // namespace
} // namespace ffmpeg
} // namespace torchaudio
