#include <libtorchaudio/rnnt/compute.h>

namespace torchaudio {


TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::rnnt_loss_forward", &rnnt_loss);
}

} // namespace torchaudio
