#include <libtorchaudio/rnnt/compute.h>

namespace torchaudio {
namespace rnnt {



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
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::rnnt_loss_forward", &rnnt_loss);
}

} // namespace torchaudio
