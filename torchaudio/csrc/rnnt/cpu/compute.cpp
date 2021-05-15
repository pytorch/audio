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
  TORCH_CHECK(
      logits.device().type() == targets.device().type(),
      "logits and targets must be on the same device");
  TORCH_CHECK(
      logits.device().type() == src_lengths.device().type(),
      "logits and logit_lengths must be on the same device");
  TORCH_CHECK(
      logits.device().type() == tgt_lengths.device().type(),
      "logits and target_lengths must be on the same device");

  TORCH_CHECK(
      logits.dtype() == torch::kFloat32 || logits.dtype() == torch::kFloat16,
      "logits must be float32 or float16 (half) type");
  TORCH_CHECK(targets.dtype() == torch::kInt32, "targets must be int32 type");
  TORCH_CHECK(
      src_lengths.dtype() == torch::kInt32, "logit_lengths must be int32 type");
  TORCH_CHECK(
      tgt_lengths.dtype() == torch::kInt32,
      "target_lengths must be int32 type");

  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
  TORCH_CHECK(src_lengths.is_contiguous(), "logit_lengths must be contiguous");
  TORCH_CHECK(tgt_lengths.is_contiguous(), "target_lengths must be contiguous");

  TORCH_CHECK(
      logits.dim() == 4, "logits must be 4-D (batch, time, target, class)");
  TORCH_CHECK(
      targets.dim() == 2, "targets must be 2-D (batch, max target length)");
  TORCH_CHECK(src_lengths.dim() == 1, "logit_lengths must be 1-D");
  TORCH_CHECK(tgt_lengths.dim() == 1, "target_lengths must be 1-D");

  TORCH_CHECK(
      src_lengths.size(0) == logits.size(0),
      "batch dimension mismatch between logits and logit_lengths");
  TORCH_CHECK(
      tgt_lengths.size(0) == logits.size(0),
      "batch dimension mismatch between logits and target_lengths");
  TORCH_CHECK(
      targets.size(0) == logits.size(0),
      "batch dimension mismatch between logits and targets");

  TORCH_CHECK(
      blank >= 0 && blank < logits.size(-1),
      "blank must be within [0, logits.shape[-1])");

  TORCH_CHECK(
      logits.size(1) == at::max(src_lengths).item().toInt(),
      "input length mismatch");
  TORCH_CHECK(
      logits.size(2) == at::max(tgt_lengths).item().toInt() + 1,
      "output length mismatch");
  TORCH_CHECK(
      targets.size(1) == at::max(tgt_lengths).item().toInt(),
      "target length mismatch");

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
      torch::TensorOptions()
          .device(logits.device())
          .dtype(torch::ScalarType::Int));

  torch::Tensor float_workspace = torch::empty(
      DtypeWorkspace<float>::ComputeSizeFromOptions(options),
      torch::TensorOptions()
          .device(logits.device())
          .dtype(torch::ScalarType::Float));

  Workspace<float> workspace(
      /*options=*/options,
      /*dtype_data=*/float_workspace.data_ptr<float>(),
      /*dtype_size=*/float_workspace.numel(),
      /*int_data=*/int_workspace.data_ptr<int>(),
      /*int_size=*/int_workspace.numel());

  switch (logits.scalar_type()) {
    case torch::ScalarType::Float: {
      Compute</*DTYPE=*/float, /*CAST_DTYPE=*/float>(
          /*workspace=*/workspace,
          /*logits=*/logits.data_ptr<float>(),
          /*targets=*/targets.data_ptr<int>(),
          /*src_lengths=*/src_lengths.data_ptr<int>(),
          /*tgt_lengths=*/tgt_lengths.data_ptr<int>(),
          /*costs=*/costs.data_ptr<float>(),
          /*gradients=*/
          (gradients == c10::nullopt) ? nullptr : gradients->data_ptr<float>());
      break;
    }
    case torch::ScalarType::Half: {
      Compute</*DTYPE=*/c10::Half, /*CAST_DTYPE=*/float>(
          /*workspace=*/workspace,
          /*logits=*/logits.data_ptr<c10::Half>(),
          /*targets=*/targets.data_ptr<int>(),
          /*src_lengths=*/src_lengths.data_ptr<int>(),
          /*tgt_lengths=*/tgt_lengths.data_ptr<int>(),
          /*costs=*/costs.data_ptr<c10::Half>(),
          /*gradients=*/
          (gradients == c10::nullopt) ? nullptr
                                      : gradients->data_ptr<c10::Half>());
      break;
    }
    default: {
      break;
    }
  };

  return std::make_tuple(costs, gradients);
}

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("rnnt_loss", &compute);
}

} // namespace cpu
} // namespace rnnt
} // namespace torchaudio
