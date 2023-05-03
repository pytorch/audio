#include "torchaudio/csrc/ffmpeg/stream_writer/stream_writer_custom_io.h"

namespace torchaudio::io {
AVIOContext* get_io_context(
    void* opaque,
    int buffer_size,
    int (*write_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence)) {
  unsigned char* buffer = static_cast<unsigned char*>(av_malloc(buffer_size));
  TORCH_CHECK(buffer, "Failed to allocate buffer.");
  AVIOContext* io_ctx = avio_alloc_context(
      buffer, buffer_size, 1, opaque, nullptr, write_packet, seek);
  if (!io_ctx) {
    av_freep(&buffer);
    TORCH_CHECK(false, "Failed to allocate AVIOContext.");
  }
  return io_ctx;
}

CustomIO::CustomIO(
    void* opaque,
    int buffer_size,
    int (*write_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence))
    : io_ctx(get_io_context(opaque, buffer_size, write_packet, seek)) {}

StreamWriterCustomIO::StreamWriterCustomIO(
    void* opaque,
    const c10::optional<std::string>& format,
    int buffer_size,
    int (*write_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence))
    : CustomIO(opaque, buffer_size, write_packet, seek),
      StreamWriter(io_ctx, format) {}
} // namespace torchaudio::io
