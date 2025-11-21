#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "forced_align(Tensor log_probs,"
      "Tensor targets,"
      "Tensor input_lengths,"
      "Tensor target_lengths,"
      "int blank) -> (Tensor, Tensor)");
}
