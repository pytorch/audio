#include <torchaudio/csrc/ffmpeg/pybind/stream_writer.h>

namespace torchaudio {
namespace ffmpeg {

StreamWriterFileObj::StreamWriterFileObj(
    py::object fileobj_,
    const c10::optional<std::string>& format,
    int64_t buffer_size)
    : FileObj(fileobj_, static_cast<int>(buffer_size), true),
      StreamWriterBinding(get_output_format_context(
          static_cast<std::string>(py::str(fileobj_.attr("__str__")())),
          format,
          pAVIO)) {}

void StreamWriterFileObj::set_metadata(
    const std::map<std::string, std::string>& metadata) {
  StreamWriter::set_metadata(map2dict(metadata));
}

void StreamWriterFileObj::add_audio_stream(
    int64_t sample_rate,
    int64_t num_channels,
    std::string format,
    const c10::optional<std::string>& encoder,
    const c10::optional<std::map<std::string, std::string>>& encoder_option,
    const c10::optional<std::string>& encoder_format) {
  StreamWriter::add_audio_stream(
      sample_rate,
      num_channels,
      format,
      encoder,
      map2dict(encoder_option),
      encoder_format);
}

void StreamWriterFileObj::add_video_stream(
    double frame_rate,
    int64_t width,
    int64_t height,
    std::string format,
    const c10::optional<std::string>& encoder,
    const c10::optional<std::map<std::string, std::string>>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<std::string>& hw_accel) {
  StreamWriter::add_video_stream(
      frame_rate,
      width,
      height,
      format,
      encoder,
      map2dict(encoder_option),
      encoder_format,
      hw_accel);
}

} // namespace ffmpeg
} // namespace torchaudio
