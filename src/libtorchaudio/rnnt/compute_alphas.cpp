#include <torch/script.h>

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "rnnt_loss_alphas(Tensor logits,"
      "Tensor targets,"
      "Tensor logit_lengths,"
      "Tensor target_lengths,"
      "int blank,"
      "float clamp) -> Tensor");
}
