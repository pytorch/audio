#pragma once
#include <torchaudio/csrc/ffmpeg/pybind/typedefs.h>
#include <torchaudio/csrc/ffmpeg/stream_reader_wrapper.h>

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
      const c10::optional<OptionDict>& option,
      int64_t buffer_size);

  std::map<std::string, std::string> get_metadata() const;
};

} // namespace ffmpeg
} // namespace torchaudio
