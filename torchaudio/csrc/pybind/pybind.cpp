#include <torch/extension.h>
#include <torchaudio/csrc/utils.h>

namespace torchaudio {
namespace {

PYBIND11_MODULE(_torchaudio, m) {
  m.def("is_kaldi_available", &is_kaldi_available, "");
  m.def("cuda_version", &cuda_version, "");
}

} // namespace
} // namespace torchaudio
