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

// Entry point into RNNT Loss
std::tuple<torch::Tensor, c10::optional<torch::Tensor>> compute(
    torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& logit_lengths,
    const torch::Tensor& target_lengths,
    int64_t blank,
    double clamp) {
  TORCH_CHECK(
      logits.device().type() == targets.device().type(),
      "logits and targets must be on the same device");
  TORCH_CHECK(
      logits.device().type() == logit_lengths.device().type(),
      "logits and logit_lengths must be on the same device");
  TORCH_CHECK(
      logits.device().type() == target_lengths.device().type(),
      "logits and target_lengths must be on the same device");

  TORCH_CHECK(
      logits.dtype() == torch::kFloat32 || logits.dtype() == torch::kFloat16,
      "logits must be float32 or float16 (half) type");
  TORCH_CHECK(targets.dtype() == torch::kInt32, "targets must be int32 type");
  TORCH_CHECK(
      logit_lengths.dtype() == torch::kInt32,
      "logit_lengths must be int32 type");
  TORCH_CHECK(
      target_lengths.dtype() == torch::kInt32,
      "target_lengths must be int32 type");

  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
  TORCH_CHECK(
      logit_lengths.is_contiguous(), "logit_lengths must be contiguous");
  TORCH_CHECK(
      target_lengths.is_contiguous(), "target_lengths must be contiguous");

  TORCH_CHECK(
      logits.dim() == 4, "logits must be 4-D (batch, time, target, class)");
  TORCH_CHECK(
      targets.dim() == 2, "targets must be 2-D (batch, max target length)");
  TORCH_CHECK(logit_lengths.dim() == 1, "logit_lengths must be 1-D");
  TORCH_CHECK(target_lengths.dim() == 1, "target_lengths must be 1-D");

  TORCH_CHECK(
      logit_lengths.size(0) == logits.size(0),
      "batch dimension mismatch between logits and logit_lengths");
  TORCH_CHECK(
      target_lengths.size(0) == logits.size(0),
      "batch dimension mismatch between logits and target_lengths");
  TORCH_CHECK(
      targets.size(0) == logits.size(0),
      "batch dimension mismatch between logits and targets");

  TORCH_CHECK(
      blank >= 0 && blank < logits.size(-1),
      "blank must be within [0, logits.shape[-1])");

  TORCH_CHECK(
      logits.size(1) == at::max(logit_lengths).item().toInt(),
      "input length mismatch");
  TORCH_CHECK(
      logits.size(2) == at::max(target_lengths).item().toInt() + 1,
      "output length mismatch");
  TORCH_CHECK(
      targets.size(1) == at::max(target_lengths).item().toInt(),
      "target length mismatch");

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
      options.batchSize_ * options.nHypos_,
      torch::TensorOptions().device(logits.device()).dtype(logits.dtype()));
  c10::optional<torch::Tensor> gradients = torch::zeros_like(logits);

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
          /*logit_lengths=*/logit_lengths.data_ptr<int>(),
          /*target_lengths=*/target_lengths.data_ptr<int>(),
          /*costs=*/costs.data_ptr<float>(),
          /*gradients=*/gradients->data_ptr<float>());
      break;
    }
    case torch::ScalarType::Half: {
      Compute</*DTYPE=*/c10::Half, /*CAST_DTYPE=*/float>(
          /*workspace=*/workspace,
          /*logits=*/logits.data_ptr<c10::Half>(),
          /*targets=*/targets.data_ptr<int>(),
          /*logit_lengths=*/logit_lengths.data_ptr<int>(),
          /*target_lengths=*/target_lengths.data_ptr<int>(),
          /*costs=*/costs.data_ptr<c10::Half>(),
          /*gradients=*/gradients->data_ptr<c10::Half>());
      break;
    }
    default: {
      break;
    }
  };

  return std::make_tuple(costs, gradients);
}

TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
  m.impl("rnnt_loss", &compute);
}

} // namespace gpu
} // namespace rnnt
} // namespace torchaudio
