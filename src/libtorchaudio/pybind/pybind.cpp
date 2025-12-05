#include <libtorchaudio/utils.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace torchaudio {
namespace {

PYBIND11_MODULE(_torchaudio, m) {
  m.def("is_align_available", &is_align_available, "");
  m.def("cuda_version", &cuda_version, "");
}

} // namespace
} // namespace torchaudio
