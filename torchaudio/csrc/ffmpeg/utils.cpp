#include <torch/script.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace ffmpeg {
namespace {

c10::Dict<std::string, std::tuple<int64_t, int64_t, int64_t>> get_versions() {
  c10::Dict<std::string, std::tuple<int64_t, int64_t, int64_t>> ret;
  ret.insert(
      "libavutil",
      std::make_tuple<>(
          LIBAVUTIL_VERSION_MAJOR,
          LIBAVUTIL_VERSION_MINOR,
          LIBAVUTIL_VERSION_MICRO));
  ret.insert(
      "libavcodec",
      std::make_tuple<>(
          LIBAVCODEC_VERSION_MAJOR,
          LIBAVCODEC_VERSION_MINOR,
          LIBAVCODEC_VERSION_MICRO));
  ret.insert(
      "libavformat",
      std::make_tuple<>(
          LIBAVFORMAT_VERSION_MAJOR,
          LIBAVFORMAT_VERSION_MINOR,
          LIBAVFORMAT_VERSION_MICRO));
  ret.insert(
      "libavfilter",
      std::make_tuple<>(
          LIBAVFILTER_VERSION_MAJOR,
          LIBAVFILTER_VERSION_MINOR,
          LIBAVFILTER_VERSION_MICRO));
  ret.insert(
      "libavdevice",
      std::make_tuple<>(
          LIBAVDEVICE_VERSION_MAJOR,
          LIBAVDEVICE_VERSION_MINOR,
          LIBAVDEVICE_VERSION_MICRO));
  return ret;
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::ffmpeg_get_versions", &get_versions);
}

} // namespace
} // namespace ffmpeg
} // namespace torchaudio
