#include <torchaudio/csrc/ffmpeg/pybind/stream_writer.h>

namespace torchaudio {
namespace io {

StreamWriterFileObj::StreamWriterFileObj(
    py::object fileobj_,
    const c10::optional<std::string>& format,
    int64_t buffer_size)
    : FileObj(fileobj_, static_cast<int>(buffer_size), true),
      StreamWriter(pAVIO, format) {}

} // namespace io
} // namespace torchaudio
