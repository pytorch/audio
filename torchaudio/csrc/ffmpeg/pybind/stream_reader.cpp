#include <torchaudio/csrc/ffmpeg/pybind/stream_reader.h>
#include <torchaudio/csrc/ffmpeg/pybind/typedefs.h>

namespace torchaudio {
namespace io {
namespace {
SrcInfoPyBind convert_pybind(SrcStreamInfo ssi) {
  return SrcInfoPyBind(std::forward_as_tuple(
      av_get_media_type_string(ssi.media_type),
      ssi.codec_name,
      ssi.codec_long_name,
      ssi.fmt_name,
      ssi.bit_rate,
      ssi.num_frames,
      ssi.bits_per_sample,
      ssi.metadata,
      ssi.sample_rate,
      ssi.num_channels,
      ssi.width,
      ssi.height,
      ssi.frame_rate));
}
} // namespace

StreamReaderFileObj::StreamReaderFileObj(
    py::object fileobj_,
    const c10::optional<std::string>& format,
    const c10::optional<std::map<std::string, std::string>>& option,
    int64_t buffer_size)
    : FileObj(fileobj_, static_cast<int>(buffer_size), false),
      StreamReaderBinding(pAVIO, format, option) {}

SrcInfoPyBind StreamReaderFileObj::get_src_stream_info(int64_t i) {
  return convert_pybind(StreamReader::get_src_stream_info(static_cast<int>(i)));
}

} // namespace io
} // namespace torchaudio
