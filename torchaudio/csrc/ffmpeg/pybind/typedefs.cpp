#include <torchaudio/csrc/ffmpeg/pybind/typedefs.h>

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
    TORCH_CHECK(
        chunk_len <= request,
        "Requested up to ",
        request,
        " bytes but, received ",
        chunk_len,
        " bytes. The given object does not confirm to read protocol of file object.");
    memcpy(buf, chunk.data(), chunk_len);
    buf += chunk_len;
    num_read += static_cast<int>(chunk_len);
  }
  return num_read == 0 ? AVERROR_EOF : num_read;
}

static int write_function(void* opaque, uint8_t* buf, int buf_size) {
  FileObj* fileobj = static_cast<FileObj*>(opaque);
  buf_size = FFMIN(buf_size, fileobj->buffer_size);

  py::bytes b(reinterpret_cast<const char*>(buf), buf_size);
  // TODO: check the return value to check
  fileobj->fileobj.attr("write")(b);
  return buf_size;
}

static int64_t seek_function(void* opaque, int64_t offset, int whence) {
  // We do not know the file size.
  if (whence == AVSEEK_SIZE) {
    return AVERROR(EIO);
  }
  FileObj* fileobj = static_cast<FileObj*>(opaque);
  return py::cast<int64_t>(fileobj->fileobj.attr("seek")(offset, whence));
}

AVIOContextPtr get_io_context(FileObj* opaque, int buffer_size, bool writable) {
  if (writable) {
    TORCH_CHECK(
        py::hasattr(opaque->fileobj, "write"),
        "`write` method is not available.");
  } else {
    TORCH_CHECK(
        py::hasattr(opaque->fileobj, "read"),
        "`read` method is not available.");
  }

  uint8_t* buffer = static_cast<uint8_t*>(av_malloc(buffer_size));
  TORCH_CHECK(buffer, "Failed to allocate buffer.");

  // If avio_alloc_context succeeds, then buffer will be cleaned up by
  // AVIOContextPtr destructor.
  // If avio_alloc_context fails, we need to clean up by ourselves.
  AVIOContext* av_io_ctx = avio_alloc_context(
      buffer,
      buffer_size,
      writable ? 1 : 0,
      static_cast<void*>(opaque),
      &read_function,
      writable ? &write_function : nullptr,
      py::hasattr(opaque->fileobj, "seek") ? &seek_function : nullptr);

  if (!av_io_ctx) {
    av_freep(&buffer);
    TORCH_CHECK(false, "Failed to allocate AVIO context.");
  }
  return AVIOContextPtr{av_io_ctx};
}
} // namespace

FileObj::FileObj(py::object fileobj_, int buffer_size, bool writable)
    : fileobj(fileobj_),
      buffer_size(buffer_size),
      pAVIO(get_io_context(this, buffer_size, writable)) {}

OptionDict map2dict(const OptionMap& src) {
  OptionDict dict;
  for (const auto& it : src) {
    dict.insert(it.first.c_str(), it.second.c_str());
  }
  return dict;
}

c10::optional<OptionDict> map2dict(const c10::optional<OptionMap>& src) {
  if (src) {
    return c10::optional<OptionDict>{map2dict(src.value())};
  }
  return {};
}

OptionMap dict2map(const OptionDict& src) {
  OptionMap ret;
  for (const auto& it : src) {
    ret.insert({it.key(), it.value()});
  }
  return ret;
}

} // namespace ffmpeg
} // namespace torchaudio
