#include <torch/script.h>
#include <torchaudio/csrc/rnnt/compute.h>

namespace torchaudio {
namespace rnnt {

class RNNTLossFunction : public torch::autograd::Function<RNNTLossFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor& logits,
      const torch::Tensor& targets,
      const torch::Tensor& src_lengths,
      const torch::Tensor& tgt_lengths,
      int64_t blank,
      double clamp,
      bool fused_log_smax = true,
      bool reuse_logits_for_grads = true) {
    torch::Tensor undef;
    auto result = rnnt_loss(
        logits,
        targets,
        src_lengths,
        tgt_lengths,
        blank,
        clamp,
        fused_log_smax,
        reuse_logits_for_grads);
    auto costs = std::get<0>(result);
    auto grads = std::get<1>(result).value_or(undef);
    ctx->save_for_backward({grads});
    return {costs, grads};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto grad = saved[0];
    auto grad_out = grad_outputs[0].view({-1, 1, 1, 1});
    auto result = grad * grad_out;
    torch::Tensor undef;
    return {result, undef, undef, undef, undef, undef, undef, undef};
  }
};

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> rnnt_loss_autograd(
    torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& src_lengths,
    const torch::Tensor& tgt_lengths,
    int64_t blank,
    double clamp,
    bool fused_log_smax = true,
    bool reuse_logits_for_grads = true) {
  at::AutoDispatchBelowADInplaceOrView guard;
  auto results = RNNTLossFunction::apply(
      logits,
      targets,
      src_lengths,
      tgt_lengths,
      blank,
      clamp,
      fused_log_smax,
      reuse_logits_for_grads);
  return std::make_tuple(results[0], results[1]);
}

TORCH_LIBRARY_IMPL(torchaudio, Autograd, m) {
  m.impl("rnnt_loss", rnnt_loss_autograd);
}

} // namespace rnnt
} // namespace torchaudio
