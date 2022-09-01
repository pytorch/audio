#pragma once
#include <torchaudio/csrc/ffmpeg/stream_writer/stream_writer.h>

namespace torchaudio {
namespace ffmpeg {

// create format context for writing media
AVFormatOutputContextPtr get_output_format_context(
    const std::string& dst,
    const c10::optional<std::string>& format,
    AVIOContext* io_ctx = nullptr);

class StreamWriterBinding : public StreamWriter,
                            public torch::CustomClassHolder {
 public:
  explicit StreamWriterBinding(AVFormatOutputContextPtr&& p);
};

} // namespace ffmpeg
} // namespace torchaudio
