#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>
#ifdef __HIP_PLATFORM_AMD__
#include <torchaudio/csrc/rnnt/hip/gpu_transducer_hip.h>
#else
#include <torchaudio/csrc/rnnt/gpu/gpu_transducer.h>
#endif

namespace torchaudio {
namespace rnnt {
namespace gpu {

torch::Tensor compute_betas(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& logit_lengths,
    const torch::Tensor& target_lengths,
    int64_t blank,
    double clamp) {
  Options options;
  options.batchSize_ = logit_lengths.size(0);
  options.nHypos_ = target_lengths.size(0) / logit_lengths.size(0);
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
      target_lengths.size(0),
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
      /*dtype_data=*/float_workspace.data_ptr<float>(),
      /*dtype_size=*/float_workspace.numel(),
      /*int_data=*/int_workspace.data_ptr<int>(),
      /*int_size=*/int_workspace.numel());

  // Only support float, this is mainly to enable easy
  // unit-testing
  ComputeBetas</*DTYPE=*/float, /*CAST_DTYPE=*/float>(
      /*workspace=*/workspace,
      /*logits=*/logits.data_ptr<float>(),
      /*targets=*/targets.data_ptr<int>(),
      /*logit_lengths=*/logit_lengths.data_ptr<int>(),
      /*target_lengths=*/target_lengths.data_ptr<int>(),
      /*costs=*/costs.data_ptr<float>(),
      /*betas=*/betas.data_ptr<float>());
  return betas;
}

TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
  m.impl("rnnt_loss_betas", &compute_betas);
}

} // namespace gpu
} // namespace rnnt
} // namespace torchaudio
