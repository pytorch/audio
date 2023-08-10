#include <torch/extension.h>
#include <torchaudio/csrc/utils.h>

namespace torchaudio {
namespace {

PYBIND11_MODULE(_torchaudio, m) {
  m.def("is_rir_available", &is_rir_available, "");
  m.def("is_align_available", &is_align_available, "");
  m.def("cuda_version", &cuda_version, "");
  m.def("find_avutil", &find_avutil, "");
}

} // namespace
} // namespace torchaudio
