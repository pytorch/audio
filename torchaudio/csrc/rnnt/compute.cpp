#include <torch/script.h>
#include <torchaudio/csrc/rnnt/compute.h>

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> rnnt_loss(
    torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& logit_lengths,
    const torch::Tensor& target_lengths,
    int64_t blank,
    double clamp,
    bool fused_log_softmax = true) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torchaudio::rnnt_loss", "")
                       .typed<decltype(rnnt_loss)>();
  return op.call(
      logits,
      targets,
      logit_lengths,
      target_lengths,
      blank,
      clamp,
      fused_log_softmax);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "rnnt_loss(Tensor logits,"
      "Tensor targets,"
      "Tensor logit_lengths,"
      "Tensor target_lengths,"
      "int blank,"
      "float clamp,"
      "bool fused_log_softmax) -> (Tensor, Tensor?)");
}
