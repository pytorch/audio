#include <torch/script.h>
#include <torchaudio/csrc/utils.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

namespace torchaudio {

bool is_kaldi_available() {
#ifdef INCLUDE_KALDI
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
