#include <torchaudio/csrc/ffmpeg/pybind/stream_reader.h>

namespace torchaudio {
namespace ffmpeg {

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

std::map<std::string, std::string> StreamReaderFileObj::get_metadata() const {
  std::map<std::string, std::string> ret;
  for (const auto& it : StreamReader::get_metadata()) {
    ret.insert({it.key(), it.value()});
  }
  return ret;
};
} // namespace ffmpeg
} // namespace torchaudio
