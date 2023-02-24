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
};

} // namespace io
} // namespace torchaudio
