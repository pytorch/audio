#include <torchaudio/csrc/ffmpeg/stream_writer/stream_writer_wrapper.h>

namespace torchaudio {
namespace ffmpeg {

AVFormatOutputContextPtr get_output_format_context(
    const std::string& dst,
    const c10::optional<std::string>& format,
    AVIOContext* io_ctx) {
  if (io_ctx) {
    TORCH_CHECK(
        format,
        "`format` must be provided when the input is file-like object.");
  }

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

  if (io_ctx) {
    p->pb = io_ctx;
    p->flags |= AVFMT_FLAG_CUSTOM_IO;
  }

  return AVFormatOutputContextPtr(p);
}

StreamWriterBinding::StreamWriterBinding(AVFormatOutputContextPtr&& p)
    : StreamWriter(std::move(p)) {}

} // namespace ffmpeg
} // namespace torchaudio
