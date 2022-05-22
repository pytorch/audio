#pragma once
#include <torch/extension.h>
#include <torchaudio/csrc/ffmpeg/stream_reader_wrapper.h>

namespace torchaudio {
namespace ffmpeg {

struct FileObj {
  py::object fileobj;
  int buffer_size;
  AVIOContextPtr pAVIO;
  FileObj(py::object fileobj, int buffer_size);
};

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
};

} // namespace ffmpeg
} // namespace torchaudio
