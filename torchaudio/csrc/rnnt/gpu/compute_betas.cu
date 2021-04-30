#include <THC/THC.h>
#include <torch/script.h>
#include <torchaudio/csrc/rnnt/gpu/gpu_transducer.h>

namespace torchaudio {
namespace rnnt {
namespace gpu {

torch::Tensor compute_betas(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& src_lengths,
    const torch::Tensor& tgt_lengths,
    int64_t blank,
    double clamp) {
  Options options;
  options.batchSize_ = src_lengths.size(0);
  options.nHypos_ = tgt_lengths.size(0) / src_lengths.size(0);
  options.maxSrcLen_ = logits.size(1);
  options.maxTgtLen_ = logits.size(2);
  options.numTargets_ = logits.size(3);
  options.blank_ = blank;
  options.clamp_ = clamp;

  CHECK_EQ(logits.device().type(), torch::DeviceType::CUDA);
  options.stream_ = at::cuda::getCurrentCUDAStream();
  cudaSetDevice(logits.get_device());
  options.device_ = GPU;

  torch::Tensor costs = torch::empty(
      tgt_lengths.size(0),
      torch::TensorOptions().device(logits.device()).dtype(logits.dtype()));

  torch::Tensor betas = torch::zeros(
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
  ComputeBetas</*DTYPE=*/float, /*CAST_DTYPE=*/float>(
      /*workspace=*/workspace,
      /*logits=*/logits.data<float>(),
      /*targets=*/targets.data<int>(),
      /*src_lengths=*/src_lengths.data<int>(),
      /*tgt_lengths=*/tgt_lengths.data<int>(),
      /*costs=*/costs.data<float>(),
      /*betas=*/betas.data<float>());
  return betas;
}

TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
  m.impl("rnnt_loss_betas", &compute_betas);
}

}  // namespace gpu
} // namespace rnnt
} // namespace torchaudio
