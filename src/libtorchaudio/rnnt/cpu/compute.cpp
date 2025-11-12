#include <libtorchaudio/rnnt/cpu/cpu_transducer.h>
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <torch/script.h>

namespace torchaudio {
namespace rnnt {
namespace cpu {

using at::ScalarType;
using at::Tensor;

// Entry point into RNNT Loss
std::tuple<Tensor, Tensor> compute(
    const Tensor& logits,
    const Tensor& targets,
    const Tensor& logit_lengths,
    const Tensor& target_lengths,
    int64_t blank,
    double clamp,
    bool fused_log_softmax = true) {
  TORCH_CHECK(logits.device().is_cpu(), "logits must be on CPU");
  TORCH_CHECK(targets.device().is_cpu(), "targets must be on CPU");
  TORCH_CHECK(logit_lengths.device().is_cpu(), "logit_lengths must be on CPU");
  TORCH_CHECK(target_lengths.device().is_cpu(), "target_lengths must be on CPU");

  TORCH_CHECK(
      logits.scalar_type() == ScalarType::Float ||
      logits.scalar_type() == ScalarType::Half,
      "logits must be float32 or float16");

  TORCH_CHECK(targets.scalar_type() == ScalarType::Int, "targets must be int32");
  TORCH_CHECK(logit_lengths.scalar_type() == ScalarType::Int, "logit_lengths must be int32");
  TORCH_CHECK(target_lengths.scalar_type() == ScalarType::Int, "target_lengths must be int32");

  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
  TORCH_CHECK(logit_lengths.is_contiguous(), "logit_lengths must be contiguous");
  TORCH_CHECK(target_lengths.is_contiguous(), "target_lengths must be contiguous");

  TORCH_CHECK(logits.dim() == 4, "logits must be 4-D");
  TORCH_CHECK(targets.dim() == 2, "targets must be 2-D");
  TORCH_CHECK(logit_lengths.dim() == 1, "logit_lengths must be 1-D");
  TORCH_CHECK(target_lengths.dim() == 1, "target_lengths must be 1-D");

  TORCH_CHECK(blank >= 0 && blank < logits.size(-1), "blank index out of range");

  auto max_ivalue = const Tensor& t {
    return t.max().item<int32_t>();
  };

  TORCH_CHECK(logits.size(1) == max_ivalue(logit_lengths), "input length mismatch");
  TORCH_CHECK(logits.size(2) == max_ivalue(target_lengths) + 1, "output length mismatch");
  TORCH_CHECK(targets.size(1) + 1 == logits.size(2), "target length mismatch");

  Options options;
  options.batchSize_ = logit_lengths.size(0);
  options.nHypos_ = target_lengths.size(0) / logit_lengths.size(0);
  options.maxSrcLen_ = logits.size(1);
  options.maxTgtLen_ = logits.size(2);
  options.numTargets_ = logits.size(3);
  options.blank_ = blank;
  options.clamp_ = clamp;
  options.fusedLogSmax_ = fused_log_softmax;
  options.device_ = CPU;

  Tensor costs = at::empty({options.batchSize_ * options.nHypos_}, logits.options());
  Tensor gradients = at::zeros_like(logits);

  Tensor int_workspace = at::empty({IntWorkspace::ComputeSizeFromOptions(options)}, logits.options().dtype(ScalarType::Int));
  Tensor float_workspace = at::empty({DtypeWorkspace<float>::ComputeSizeFromOptions(options)}, logits.options().dtype(ScalarType::Float));

  Workspace<float> workspace(
      options,
      reinterpret_cast<float*>(float_workspace.data_ptr()),
      float_workspace.numel(),
      reinterpret_cast<int*>(int_workspace.data_ptr()),
      int_workspace.numel());

  switch (logits.scalar_type()) {
    case ScalarType::Float: {
      Compute<float, float>(
                   reinterpret_cast<float*>(logits.data_ptr()),
          reinterpret_cast<int*>(targets.data_ptr()),
          reinterpret_cast<int*>(logit_lengths.data_ptr()),
          reinterpret_cast<int*>(target_lengths.data_ptr()),
          reinterpret_cast<float*>(costs.data_ptr()),
          reinterpret_cast<float*>(gradients.data_ptr()));
      break;
    }
    case ScalarType::Half: {
      Compute<c10::Half, float>(
          workspace,
          reinterpret_cast<c10::Half*>(logits.data_ptr()),
          reinterpret_cast<int*>(targets.data_ptr()),
          reinterpret_cast<int*>(logit_lengths.data_ptr()),
          reinterpret_cast<int*>(target_lengths.data_ptr()),
          reinterpret_cast<c10::Half*>(costs.data_ptr()),
          reinterpret_cast<c10::Half*>(gradients.data_ptr()));
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported dtype");
  }

  return std::make_tuple(costs, gradients);
}

void boxed_rnnt_loss(c10::IValue* stack, uint64_t num_args, uint64_t num_outputs) {
  TORCH_CHECK(num_args == 7, "num_args must be 7");
  TORCH_CHECK(num_outputs == 2, "num_outputs must be 2");

  auto logits = stack[0].toTensor();
  auto targets = stack[1].toTensor();
  auto logit_lengths = stack[2].toTensor();
  auto target_lengths = stack[3].toTensor();
  auto blank = stack[4].toInt();
  auto clamp = stack[5].toDouble();
  auto fused_log_softmax = stack[6].toBool();

  auto res = compute(logits, targets, logit_lengths, target_lengths, blank, clamp, fused_log_softmax);

  stack[0] = c10::IValue(std::get<0>(res));
  stack[1] = c10::IValue(std::get<1>(res));
}

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("rnnt_loss_forward", &boxed_rnnt_loss);
}

} // namespace cpu
} // namespace rnnt
} // namespace torchaudio
