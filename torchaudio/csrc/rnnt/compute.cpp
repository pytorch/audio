#include <torch/script.h>
#include <torchaudio/csrc/rnnt/compute.h>

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> rnnt_loss(
    torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& src_lengths,
    const torch::Tensor& tgt_lengths,
    int64_t blank,
    double clamp,
    bool fused_log_smax = true,
    bool reuse_logits_for_grads = true) {
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("torchaudio::rnnt_loss", "")
    .typed<decltype(rnnt_loss)>();
  return op.call(logits, targets, src_lengths, tgt_lengths, blank, clamp,
                 fused_log_smax, reuse_logits_for_grads);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "rnnt_loss(Tensor logits,"
      "Tensor targets,"
      "Tensor src_lengths,"
      "Tensor tgt_lengths,"
      "int blank,"
      "float clamp,"
      "bool fused_log_smax=True,"
      "bool reuse_logits_for_grads=True) -> (Tensor, Tensor?)");
}
