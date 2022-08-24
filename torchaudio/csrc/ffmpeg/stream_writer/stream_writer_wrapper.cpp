#include <torchaudio/csrc/ffmpeg/stream_writer/stream_writer_wrapper.h>

namespace torchaudio {
namespace ffmpeg {

AVFormatOutputContextPtr get_output_format_context(
    const std::string& dst,
    const c10::optional<std::string>& format) {
  AVFormatContext* p = avformat_alloc_context();
  TORCH_CHECK(p, "Failed to allocate AVFormatContext.");

  int ret = avformat_alloc_output_context2(
      &p, nullptr, format ? format.value().c_str() : nullptr, dst.c_str());
  TORCH_CHECK(
      ret >= 0,
      "Failed to open output \"",
      dst,
      "\" (",
      av_err2string(ret),
      ").");

  return AVFormatOutputContextPtr(p);
}

StreamWriterBinding::StreamWriterBinding(AVFormatOutputContextPtr&& p)
    : StreamWriter(std::move(p)) {}

} // namespace ffmpeg
} // namespace torchaudio
