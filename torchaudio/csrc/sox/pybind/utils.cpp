#include <torchaudio/csrc/sox/pybind/utils.h>

namespace torchaudio::sox {

auto read_fileobj(py::object* fileobj, const uint64_t size, char* buffer)
    -> uint64_t {
  uint64_t num_read = 0;
  while (num_read < size) {
    auto request = size - num_read;
    auto chunk = static_cast<std::string>(
        static_cast<py::bytes>(fileobj->attr("read")(request)));
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
    memcpy(buffer, chunk.data(), chunk_len);
    buffer += chunk_len;
    num_read += chunk_len;
  }
  return num_read;
}

} // namespace torchaudio::sox
