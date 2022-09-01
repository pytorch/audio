#include <torchaudio/csrc/ffmpeg/pybind/stream_reader.h>
#include <torchaudio/csrc/ffmpeg/pybind/typedefs.h>

namespace torchaudio {
namespace ffmpeg {
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
      dict2map(ssi.metadata),
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
      StreamReaderBinding(get_input_format_context(
          static_cast<std::string>(py::str(fileobj_.attr("__str__")())),
          format,
          map2dict(option),
          pAVIO)) {}

std::map<std::string, std::string> StreamReaderFileObj::get_metadata() const {
  return dict2map(StreamReader::get_metadata());
};

SrcInfoPyBind StreamReaderFileObj::get_src_stream_info(int64_t i) {
  return convert_pybind(StreamReader::get_src_stream_info(static_cast<int>(i)));
}

void StreamReaderFileObj::add_audio_stream(
    int64_t i,
    int64_t frames_per_chunk,
    int64_t num_chunks,
    const c10::optional<std::string>& filter_desc,
    const c10::optional<std::string>& decoder,
    const c10::optional<std::map<std::string, std::string>>& decoder_option) {
  StreamReader::add_audio_stream(
      i,
      frames_per_chunk,
      num_chunks,
      filter_desc,
      decoder,
      map2dict(decoder_option));
}

void StreamReaderFileObj::add_video_stream(
    int64_t i,
    int64_t frames_per_chunk,
    int64_t num_chunks,
    const c10::optional<std::string>& filter_desc,
    const c10::optional<std::string>& decoder,
    const c10::optional<std::map<std::string, std::string>>& decoder_option,
    const c10::optional<std::string>& hw_accel) {
  StreamReader::add_video_stream(
      i,
      frames_per_chunk,
      num_chunks,
      filter_desc,
      decoder,
      map2dict(decoder_option),
      hw_accel);
}

} // namespace ffmpeg
} // namespace torchaudio
