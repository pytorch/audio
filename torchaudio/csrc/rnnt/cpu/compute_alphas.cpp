#include <torch/script.h>
#include <torchaudio/csrc/rnnt/cpu/cpu_transducer.h>

namespace torchaudio {
namespace rnnt {
namespace cpu {

torch::Tensor compute_alphas(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& src_lengths,
    const torch::Tensor& tgt_lengths,
    int64_t blank,
    double clamp,
    const c10::optional<torch::Tensor>& wp_ends = c10::nullopt,
    int64_t l_buffer = 0,
    int64_t r_buffer = 0) {
  Options options;
  options.batchSize_ = src_lengths.size(0);
  options.nHypos_ = tgt_lengths.size(0) / src_lengths.size(0);
  options.maxSrcLen_ = logits.size(1);
  options.maxTgtLen_ = logits.size(2);
  options.numTargets_ = logits.size(3);
  options.blank_ = blank;
  options.clamp_ = clamp;
  options.lBuffer_ = l_buffer;
  options.rBuffer_ = r_buffer;

  CHECK_EQ(logits.device().type(), torch::DeviceType::CPU);
  options.device_ = CPU;

  torch::Tensor alphas = torch::zeros(
      {options.batchSize_ * options.nHypos_,
       options.maxSrcLen_,
       options.maxTgtLen_},
      torch::TensorOptions().device(logits.device()).dtype(logits.dtype()));

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
      /*dtype_data=*/float_workspace.data<float>(),
      /*dtype_size=*/float_workspace.numel(),
      /*int_data=*/int_workspace.data<int>(),
      /*int_size=*/int_workspace.numel());

  // Only support float, this is mainly to enable easy
  // unit-testing
  ComputeAlphas</*DTYPE=*/float, /*CAST_DTYPE=*/float>(
      /*workspace=*/workspace,
      /*logits=*/logits.data<float>(),
      /*targets=*/targets.data<int>(),
      /*src_lengths=*/src_lengths.data<int>(),
      /*tgt_lengths=*/tgt_lengths.data<int>(),
      /*alphas=*/alphas.data<float>(),
      /*wp_ends=*/(wp_ends == c10::nullopt) ? nullptr : wp_ends->data<int>());
  return alphas;
}

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("rnnt_loss_alphas", &compute_alphas);
}

} // namespace cpu
} // namespace rnnt
} // namespace torchaudio
