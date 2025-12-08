#include <libtorchaudio/utils.h>

// It is safe to temporarily disable TORCH_TARGET_VERSION for pybind11
// as it is a header-only library.
#ifdef TORCH_TARGET_VERSION
#define SAVE_TORCH_TARGET_VERSION TORCH_TARGET_VERSION
#undef TORCH_TARGET_VERSION
#endif
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef SAVE_TORCH_TARGET_VERSION
#define TORCH_TARGET_VERSION SAVE_TORCH_TARGET_VERSION
#undef SAVE_TORCH_TARGET_VERSION
#endif

namespace torchaudio {
namespace {

PYBIND11_MODULE(_torchaudio, m) {
  m.def("is_align_available", &is_align_available, "");
  m.def("cuda_version", &cuda_version, "");
}

} // namespace
} // namespace torchaudio
