#include <libtorchaudio/rnnt/gpu/gpu_transducer.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

namespace torchaudio {
namespace rnnt {
namespace gpu {

using torch::stable::Tensor;
using torch::headeronly::ScalarType;

// Entry point into RNNT Loss
std::tuple<Tensor, Tensor> compute(
    const Tensor& logits,
    const Tensor& targets,
    const Tensor& logit_lengths,
    const Tensor& target_lengths,
    int64_t blank,
    double clamp,
    bool fused_log_softmax = true) {
  STD_TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");

  STD_TORCH_CHECK(
      targets.is_cuda() && targets.get_device_index() == logits.get_device_index(),
      "logits and targets must be on the same device");
  STD_TORCH_CHECK(
      logit_lengths.is_cuda() && logit_lengths.get_device_index() == logits.get_device_index(),
      "logits and logit_lengths must be on the same device");
  STD_TORCH_CHECK(
      target_lengths.is_cuda() && target_lengths.get_device_index() == logits.get_device_index(),
      "logits and target_lengths must be on the same device");

  STD_TORCH_CHECK(
      logits.scalar_type() == ScalarType::Float || logits.scalar_type() == ScalarType::Half,
      "logits must be float32 or float16 (half) type");

  STD_TORCH_CHECK(targets.scalar_type() == ScalarType::Int, "targets must be int32 type");

  STD_TORCH_CHECK(
      logit_lengths.scalar_type() == ScalarType::Int,
      "logit_lengths must be int32 type");
  STD_TORCH_CHECK(
      target_lengths.scalar_type() == ScalarType::Int,
      "target_lengths must be int32 type");

  STD_TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  STD_TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
  STD_TORCH_CHECK(
      logit_lengths.is_contiguous(), "logit_lengths must be contiguous");
  STD_TORCH_CHECK(
      target_lengths.is_contiguous(), "target_lengths must be contiguous");

  STD_TORCH_CHECK(
      logits.dim() == 4, "logits must be 4-D (batch, time, target, class)");
  STD_TORCH_CHECK(
      targets.dim() == 2, "targets must be 2-D (batch, max target length)");
  STD_TORCH_CHECK(logit_lengths.dim() == 1, "logit_lengths must be 1-D");
  STD_TORCH_CHECK(target_lengths.dim() == 1, "target_lengths must be 1-D");

  STD_TORCH_CHECK(
      logit_lengths.size(0) == logits.size(0),
      "batch dimension mismatch between logits and logit_lengths");
  STD_TORCH_CHECK(
      target_lengths.size(0) == logits.size(0),
      "batch dimension mismatch between logits and target_lengths");
  STD_TORCH_CHECK(
      targets.size(0) == logits.size(0),
      "batch dimension mismatch between logits and targets");

  STD_TORCH_CHECK(
      blank >= 0 && blank < logits.size(-1),
      "blank must be within [0, logits.shape[-1])");

  auto max_ivalue = [](const Tensor& t) {
    int32_t value;
    C10_CUDA_CHECK(cudaMemcpy(&value, torch::stable::amax(t, {}).data_ptr(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    return value;
  };

  STD_TORCH_CHECK(
      logits.size(1) == max_ivalue(logit_lengths),
      "input length mismatch");
  STD_TORCH_CHECK(
      logits.size(2) == max_ivalue(target_lengths) + 1,
      "output length mismatch");
  STD_TORCH_CHECK(
      targets.size(1) + 1 == logits.size(2),
      "target length mismatch");

  Options options;
  options.batchSize_ = logit_lengths.size(0);
  options.nHypos_ = target_lengths.size(0) / logit_lengths.size(0);
  options.maxSrcLen_ = logits.size(1);
  options.maxTgtLen_ = logits.size(2);
  options.numTargets_ = logits.size(3);
  options.blank_ = blank;
  options.clamp_ = clamp;
  options.fusedLogSmax_ = fused_log_softmax;
  options.stream_ = at::cuda::getCurrentCUDAStream();
  cudaSetDevice(logits.get_device());
  options.device_ = GPU;

  Tensor costs = torch::stable::new_empty(logits, {options.batchSize_ * options.nHypos_});
  Tensor gradients = torch::stable::empty_like(logits);
  torch::stable::fill_(gradients, 0.0);

  Tensor int_workspace = torch::stable::new_empty(logits, {IntWorkspace::ComputeSizeFromOptions(options)}, ScalarType::Int);
  Tensor float_workspace = torch::stable::new_empty(logits, {DtypeWorkspace<float>::ComputeSizeFromOptions(options)}, ScalarType::Float);

  Workspace<float> workspace(
      /*options=*/options,
      /*dtype_data=*/reinterpret_cast<float*>(float_workspace.data_ptr()),
      /*dtype_size=*/float_workspace.numel(),
      /*int_data=*/reinterpret_cast<int*>(int_workspace.data_ptr()),
      /*int_size=*/int_workspace.numel());

  switch (logits.scalar_type()) {
    case ScalarType::Float: {
      Compute</*DTYPE=*/float, /*CAST_DTYPE=*/float>(
          /*workspace=*/workspace,
          /*logits=*/reinterpret_cast<float*>(logits.data_ptr()),
          /*targets=*/reinterpret_cast<int*>(targets.data_ptr()),
          /*srcLengths=*/reinterpret_cast<int*>(logit_lengths.data_ptr()),
          /*tgtLengths=*/reinterpret_cast<int*>(target_lengths.data_ptr()),
          /*costs=*/reinterpret_cast<float*>(costs.data_ptr()),
          /*gradients=*/reinterpret_cast<float*>(gradients.data_ptr()));
      break;
    }
    case ScalarType::Half: {
      Compute</*DTYPE=*/c10::Half, /*CAST_DTYPE=*/float>(
          /*workspace=*/workspace,
          /*logits=*/reinterpret_cast<c10::Half*>(logits.data_ptr()),
          /*targets=*/reinterpret_cast<int*>(targets.data_ptr()),
          /*srcLengths=*/reinterpret_cast<int*>(logit_lengths.data_ptr()),
          /*tgtLengths=*/reinterpret_cast<int*>(target_lengths.data_ptr()),
          /*costs=*/reinterpret_cast<c10::Half*>(costs.data_ptr()),
          /*gradients=*/reinterpret_cast<c10::Half*>(gradients.data_ptr()));
      break;
    }
    default: {
      STD_TORCH_CHECK(false, "unreachable");
    }
  };

  return std::make_tuple(costs, gradients);
}

void boxed_rnnt_loss(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  STD_TORCH_CHECK(num_args == 7, "num_args must be 7");
  STD_TORCH_CHECK(num_outputs == 2, "num_outputs must be 2");
  std::tuple<Tensor, Tensor> res = compute(
      /*logits*/torch::stable::detail::to<Tensor>(stack[0]),
      /*targets*/torch::stable::detail::to<Tensor>(stack[1]),
      /*logit_lengths*/torch::stable::detail::to<Tensor>(stack[2]),
      /*target_lengths*/torch::stable::detail::to<Tensor>(stack[3]),
      /*blank*/float(torch::stable::detail::to<int64_t>(stack[4])),
      /*clamp*/torch::stable::detail::to<double>(stack[5]),
      /*fused_log_softmax*/torch::stable::detail::to<bool>(stack[6]));
  stack[0] = torch::stable::detail::from(std::get<0>(res));
  stack[1] = torch::stable::detail::from(std::get<1>(res));
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
  m.impl("rnnt_loss_forward", &boxed_rnnt_loss);
}

} // namespace gpu
} // namespace rnnt
} // namespace torchaudio
