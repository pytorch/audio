#ifndef TORCHAUDIO_PYBIND_SOX_UTILS_H
#define TORCHAUDIO_PYBIND_SOX_UTILS_H

#include <torch/extension.h>

namespace torchaudio {
namespace sox_utils {

uint64_t read_fileobj(py::object* fileobj, uint64_t size, char* buffer);

} // namespace sox_utils
} // namespace torchaudio

#endif
