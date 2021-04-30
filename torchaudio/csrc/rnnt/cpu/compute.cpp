#include <torch/script.h>
#include <torchaudio/csrc/rnnt/cpu/cpu_transducer.h>

namespace torchaudio {
namespace rnnt {
namespace cpu {

// Entry point into RNNT Loss
std::tuple<torch::Tensor, c10::optional<torch::Tensor>> compute(
    torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& src_lengths,
    const torch::Tensor& tgt_lengths,
    int64_t blank,
    double clamp,
    bool fused_log_smax = true,
    bool reuse_logits_for_grads = true) {

  Options options;
  options.batchSize_ = src_lengths.size(0);
  options.nHypos_ = tgt_lengths.size(0) / src_lengths.size(0);
  options.maxSrcLen_ = logits.size(1);
  options.maxTgtLen_ = logits.size(2);
  options.numTargets_ = logits.size(3);
  options.blank_ = blank;
  options.clamp_ = clamp;
  options.fusedLogSmax_ = fused_log_smax;

  CHECK_EQ(logits.device().type(), torch::DeviceType::CPU);
  options.device_ = CPU;

  torch::Tensor costs = torch::empty(
      options.batchSize_ * options.nHypos_,
      torch::TensorOptions().device(logits.device()).dtype(logits.dtype()));
  c10::optional<torch::Tensor> gradients = c10::nullopt;
  if (logits.requires_grad()) {
    if (reuse_logits_for_grads) {
      gradients = logits;
    } else {
      gradients = torch::zeros_like(logits);
    }
  }

  torch::Tensor int_workspace = torch::empty(
      IntWorkspace::ComputeSizeFromOptions(options),
      torch::TensorOptions().device(logits.device()).dtype(torch::ScalarType::Int));

  torch::Tensor float_workspace = torch::empty(
      DtypeWorkspace<float>::ComputeSizeFromOptions(options),
      torch::TensorOptions().device(logits.device()).dtype(torch::ScalarType::Float));

  Workspace<float> workspace(
      /*options=*/options,
      /*dtype_data=*/float_workspace.data<float>(),
      /*dtype_size=*/float_workspace.numel(),
      /*int_data=*/int_workspace.data<int>(),
      /*int_size=*/int_workspace.numel());

  switch (logits.type().scalarType()) {
    case torch::ScalarType::Float:
      {
        Compute</*DTYPE=*/float, /*CAST_DTYPE=*/float>(
            /*workspace=*/workspace,
            /*logits=*/logits.data<float>(),
            /*targets=*/targets.data<int>(),
            /*src_lengths=*/src_lengths.data<int>(),
            /*tgt_lengths=*/tgt_lengths.data<int>(),
            /*costs=*/costs.data<float>(),
            /*gradients=*/(gradients == c10::nullopt)? nullptr : gradients->data<float>());
        break;
      }
    case torch::ScalarType::Half:
      {
        Compute</*DTYPE=*/c10::Half, /*CAST_DTYPE=*/float>(
            /*workspace=*/workspace,
            /*logits=*/logits.data<c10::Half>(),
            /*targets=*/targets.data<int>(),
            /*src_lengths=*/src_lengths.data<int>(),
            /*tgt_lengths=*/tgt_lengths.data<int>(),
            /*costs=*/costs.data<c10::Half>(),
            /*gradients=*/(gradients == c10::nullopt)? nullptr : gradients->data<c10::Half>());
        break;
      }
    default:
      {
        LOG(ERROR) << "unsupported logits.type().scalarType() = "
                   << logits.type().scalarType();
        break;
      }
  };

  return std::make_tuple(costs, gradients);
}

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("rnnt_loss", &compute);
}

}  // namespace cpu
}  // namespace rnnt
}  // namespace torchaudio
