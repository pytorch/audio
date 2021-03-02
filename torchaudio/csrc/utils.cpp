#include <torch/script.h>

namespace {

bool is_sox_available() {
#ifdef INCLUDE_SOX
  return true;
#else
  return false;
#endif
}

} // namespace

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::is_sox_available", &is_sox_available);
}
