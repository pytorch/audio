#pragma once
#include <torchaudio/csrc/ffmpeg/pybind/typedefs.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/stream_reader.h>

namespace torchaudio {
namespace io {

// The reason we inherit FileObj instead of making it an attribute
// is so that FileObj is instantiated first.
// AVIOContext must be initialized before AVFormat, and outlive AVFormat.
class StreamReaderFileObj : private FileObj, public StreamReader {
 public:
  StreamReaderFileObj(
      py::object fileobj,
      const c10::optional<std::string>& format,
      const c10::optional<std::map<std::string, std::string>>& option,
      int64_t buffer_size);
};

} // namespace io
} // namespace torchaudio
