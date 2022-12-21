#include <torch/extension.h>
#include <torchaudio/csrc/utils.h>

namespace torchaudio {
namespace {

// Note
// These functions are not intended for a real usecase.
// They are accessible via TorchBind.
// It is beneficial to have _torchaudio that is linked to libtorchaudio,
// when torchaudio is deployed with PEX format, where the library location
// is not in torchaudio/lib. But somewhere in LD_LIBRARY_PATH.
// In this case, attempt to import _torchaudio will automatically resolves
// libtorchaudio, if _torchaudio is linked to libtorchaudio.

PYBIND11_MODULE(_torchaudio, m) {
  m.def("is_kaldi_available", &is_kaldi_available, "");
  m.def("cuda_version", &cuda_version, "");
}

} // namespace
} // namespace torchaudio
