#include <torchaudio/csrc/ffmpeg/pybind/stream_reader.h>
#include <torchaudio/csrc/ffmpeg/pybind/typedefs.h>

namespace torchaudio {
namespace io {

StreamReaderFileObj::StreamReaderFileObj(
    py::object fileobj_,
    const c10::optional<std::string>& format,
    const c10::optional<std::map<std::string, std::string>>& option,
    int64_t buffer_size)
    : FileObj(fileobj_, static_cast<int>(buffer_size), false),
      StreamReader(pAVIO, format, option) {}

} // namespace io
} // namespace torchaudio
