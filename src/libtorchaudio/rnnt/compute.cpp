#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "torchaudio::rnnt_loss(Tensor logits,"
      "Tensor targets,"
      "Tensor logit_lengths,"
      "Tensor target_lengths,"
      "int blank,"
      "float clamp,"
      "bool fused_log_softmax) -> (Tensor, Tensor?)");
}
