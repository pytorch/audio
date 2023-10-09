#include <libtorchaudio/forced_align/compute.h>
#include <torch/script.h>

std::tuple<torch::Tensor, torch::Tensor> forced_align(
    const torch::Tensor& logProbs,
    const torch::Tensor& targets,
    const torch::Tensor& inputLengths,
    const torch::Tensor& targetLengths,
    const int64_t blank) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torchaudio::forced_align", "")
                       .typed<decltype(forced_align)>();
  return op.call(logProbs, targets, inputLengths, targetLengths, blank);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "forced_align(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank) -> (Tensor, Tensor)");
}
