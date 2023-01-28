#pragma once
#include <torchaudio/csrc/ffmpeg/pybind/typedefs.h>
#include <torchaudio/csrc/ffmpeg/stream_writer/stream_writer.h>

namespace torchaudio {
namespace io {

class StreamWriterFileObj : private FileObj, public StreamWriter {
 public:
  StreamWriterFileObj(
      py::object fileobj,
      const c10::optional<std::string>& format,
      int64_t buffer_size);

  void set_metadata(const std::map<std::string, std::string>&);
  void add_audio_stream(
      int64_t sample_rate,
      int64_t num_channels,
      std::string format,
      const c10::optional<std::string>& encoder,
      const c10::optional<std::map<std::string, std::string>>& encoder_option,
      const c10::optional<std::string>& encoder_format);
  void add_video_stream(
      double frame_rate,
      int64_t width,
      int64_t height,
      std::string format,
      const c10::optional<std::string>& encoder,
      const c10::optional<std::map<std::string, std::string>>& encoder_option,
      const c10::optional<std::string>& encoder_format,
      const c10::optional<std::string>& hw_accel);
};

} // namespace io
} // namespace torchaudio
