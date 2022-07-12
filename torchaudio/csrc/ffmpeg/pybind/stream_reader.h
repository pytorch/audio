#pragma once
#include <torchaudio/csrc/ffmpeg/pybind/typedefs.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/stream_reader_wrapper.h>

namespace torchaudio {
namespace ffmpeg {

// The reason we inherit FileObj instead of making it an attribute
// is so that FileObj is instantiated first.
// AVIOContext must be initialized before AVFormat, and outlive AVFormat.
class StreamReaderFileObj : protected FileObj, public StreamReaderBinding {
 public:
  StreamReaderFileObj(
      py::object fileobj,
      const c10::optional<std::string>& format,
      const c10::optional<std::map<std::string, std::string>>& option,
      int64_t buffer_size);

  std::map<std::string, std::string> get_metadata() const;

  SrcInfoPyBind get_src_stream_info(int64_t i);

  void add_audio_stream(
      int64_t i,
      int64_t frames_per_chunk,
      int64_t num_chunks,
      const c10::optional<std::string>& filter_desc,
      const c10::optional<std::string>& decoder,
      const c10::optional<std::map<std::string, std::string>>& decoder_option);
  void add_video_stream(
      int64_t i,
      int64_t frames_per_chunk,
      int64_t num_chunks,
      const c10::optional<std::string>& filter_desc,
      const c10::optional<std::string>& decoder,
      const c10::optional<std::map<std::string, std::string>>& decoder_option,
      const c10::optional<std::string>& hw_accel);
};

} // namespace ffmpeg
} // namespace torchaudio
