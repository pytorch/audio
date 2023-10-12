#include <ATen/DynamicLibrary.h>
#include <libtorchaudio/utils.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

namespace torchaudio {

bool is_rir_available() {
#ifdef INCLUDE_RIR
  return true;
#else
  return false;
#endif
}

bool is_align_available() {
#ifdef INCLUDE_ALIGN
  return true;
#else
  return false;
#endif
}

c10::optional<int64_t> cuda_version() {
#ifdef USE_CUDA
  return CUDA_VERSION;
#else
  return {};
#endif
}

} // namespace torchaudio
