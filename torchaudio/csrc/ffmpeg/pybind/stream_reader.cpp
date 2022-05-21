#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/pybind/stream_reader.h>

namespace torchaudio {
namespace ffmpeg {
namespace {

static int read_function(void* opaque, uint8_t* buf, int buf_size) {
  FileObj* fileobj = static_cast<FileObj*>(opaque);
  buf_size = FFMIN(buf_size, fileobj->buffer_size);

  int num_read = 0;
  while (num_read < buf_size) {
    int request = buf_size - num_read;
    auto chunk = static_cast<std::string>(
        static_cast<py::bytes>(fileobj->fileobj.attr("read")(request)));
    auto chunk_len = chunk.length();
    if (chunk_len == 0) {
      break;
    }
    if (chunk_len > request) {
      std::ostringstream message;
      message
          << "Requested up to " << request << " bytes but, "
          << "received " << chunk_len << " bytes. "
          << "The given object does not confirm to read protocol of file object.";
      throw std::runtime_error(message.str());
    }
    memcpy(buf, chunk.data(), chunk_len);
    buf += chunk_len;
    num_read += chunk_len;
  }
  return num_read == 0 ? AVERROR_EOF : num_read;
}

static int64_t seek_function(void* opaque, int64_t offset, int whence) {
  // We do not know the file size.
  if (whence == AVSEEK_SIZE) {
    return AVERROR(EIO);
  }
  FileObj* fileobj = static_cast<FileObj*>(opaque);
  return py::cast<int64_t>(fileobj->fileobj.attr("seek")(offset, whence));
}

AVIOContextPtr get_io_context(FileObj* opaque, int buffer_size) {
  uint8_t* buffer = static_cast<uint8_t*>(av_malloc(buffer_size));
  if (!buffer) {
    throw std::runtime_error("Failed to allocate buffer.");
  }

  // If avio_alloc_context succeeds, then buffer will be cleaned up by
  // AVIOContextPtr destructor.
  // If avio_alloc_context fails, we need to clean up by ourselves.
  AVIOContext* av_io_ctx = avio_alloc_context(
      buffer,
      buffer_size,
      0,
      static_cast<void*>(opaque),
      &read_function,
      nullptr,
      py::hasattr(opaque->fileobj, "seek") ? &seek_function : nullptr);

  if (!av_io_ctx) {
    av_freep(&buffer);
    throw std::runtime_error("Failed to allocate AVIO context.");
  }
  return AVIOContextPtr{av_io_ctx};
}
} // namespace

FileObj::FileObj(py::object fileobj_, int buffer_size)
    : fileobj(fileobj_),
      buffer_size(buffer_size),
      pAVIO(get_io_context(this, buffer_size)) {}

StreamReaderFileObj::StreamReaderFileObj(
    py::object fileobj_,
    const c10::optional<std::string>& format,
    const c10::optional<OptionDict>& option,
    int64_t buffer_size)
    : FileObj(fileobj_, static_cast<int>(buffer_size)),
      StreamReaderBinding(get_input_format_context(
          static_cast<std::string>(py::str(fileobj_.attr("__str__")())),
          format,
          option.value_or(OptionDict{}),
          pAVIO)) {}

} // namespace ffmpeg
} // namespace torchaudio
