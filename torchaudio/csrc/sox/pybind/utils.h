#ifndef TORCHAUDIO_PYBIND_SOX_UTILS_H
#define TORCHAUDIO_PYBIND_SOX_UTILS_H

#include <torch/extension.h>

namespace torchaudio {
namespace sox_utils {

auto read_fileobj(py::object* fileobj, uint64_t size, char* buffer) -> uint64_t;

} // namespace sox_utils
} // namespace torchaudio

#endif
