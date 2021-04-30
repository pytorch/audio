#include <THC/THC.h>
#include <torch/script.h>
#include <torchaudio/csrc/rnnt/gpu/gpu_transducer.h>

namespace torchaudio {
namespace rnnt {
namespace gpu {

// Entry point into Sparse Alignment Restricted RNNT Loss.
std::tuple<torch::Tensor, c10::optional<torch::Tensor>>
compute_sparse(
    torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& src_lengths,
    const torch::Tensor& tgt_lengths,
    int64_t blank,
    double clamp,
    int64_t max_T,
    int64_t max_U,
    const c10::optional<torch::Tensor>& wp_ends = c10::nullopt,
    int64_t l_buffer = 0,
    int64_t r_buffer = 0,
    const c10::optional<torch::Tensor>& valid_ranges = c10::nullopt,
    const c10::optional<torch::Tensor>& cells_per_sample = c10::nullopt,
    bool fused_log_smax = true,
    bool reuse_logits_for_grads = true) {

  Options options;
  options.batchSize_ = src_lengths.size(0);
  options.nHypos_ = tgt_lengths.size(0) / src_lengths.size(0);
  options.maxSrcLen_ = max_T;
  options.maxTgtLen_ = max_U;
  options.sparseCells_ = logits.size(0);
  options.numTargets_ = logits.size(1);
  options.fusedLogSmax_ = fused_log_smax;

  options.blank_ = blank;
  options.clamp_ = clamp;
  options.lBuffer_ = l_buffer;
  options.rBuffer_ = r_buffer;
  options.sparse_ = true;

  CHECK_EQ(logits.device().type(), torch::DeviceType::CUDA);
  options.stream_ = at::cuda::getCurrentCUDAStream();
  cudaSetDevice(logits.get_device());
  options.device_ = GPU;

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
        ComputeSparse</*DTYPE=*/float, /*CAST_DTYPE=*/float>(
            /*workspace=*/workspace,
            /*logits=*/logits.data<float>(),
            /*targets=*/targets.data<int>(),
            /*src_lengths=*/src_lengths.data<int>(),
            /*tgt_lengths=*/tgt_lengths.data<int>(),
            /*costs=*/costs.data<float>(),
            /*gradients=*/(gradients == c10::nullopt)? nullptr : gradients->data<float>(),
            /*wp_ends=*/(wp_ends == c10::nullopt)? nullptr : wp_ends->data<int>(),
            (valid_ranges == c10::nullopt)? nullptr : valid_ranges->data<int>(),
            (cells_per_sample == c10::nullopt)? nullptr : cells_per_sample->data<int>());
        break;
      }
    case torch::ScalarType::Half:
      {
        ComputeSparse</*DTYPE=*/c10::Half, /*CAST_DTYPE=*/float>(
            /*workspace=*/workspace,
            /*logits=*/logits.data<c10::Half>(),
            /*targets=*/targets.data<int>(),
            /*src_lengths=*/src_lengths.data<int>(),
            /*tgt_lengths=*/tgt_lengths.data<int>(),
            /*costs=*/costs.data<c10::Half>(),
            /*gradients=*/(gradients == c10::nullopt)? nullptr : gradients->data<c10::Half>(),
            /*wp_ends=*/(wp_ends == c10::nullopt)? nullptr : wp_ends->data<int>(),
            (valid_ranges == c10::nullopt)? nullptr : valid_ranges->data<int>(),
            (cells_per_sample == c10::nullopt)? nullptr : cells_per_sample->data<int>());
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


// Entry point into RNNT Loss
std::tuple<torch::Tensor, c10::optional<torch::Tensor>>
compute(
    torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& src_lengths,
    const torch::Tensor& tgt_lengths,
    int64_t blank,
    double clamp,
    const c10::optional<torch::Tensor>& wp_ends = c10::nullopt,
    int64_t l_buffer = 0,
    int64_t r_buffer = 0,
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
  options.lBuffer_ = l_buffer;
  options.rBuffer_ = r_buffer;
  options.fusedLogSmax_ = fused_log_smax;

  CHECK_EQ(logits.device().type(), torch::DeviceType::CUDA);
  options.stream_ = at::cuda::getCurrentCUDAStream();
  cudaSetDevice(logits.get_device());
  options.device_ = GPU;

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
            /*gradients=*/(gradients == c10::nullopt)? nullptr : gradients->data<float>(),
            /*wp_ends=*/(wp_ends == c10::nullopt)? nullptr : wp_ends->data<int>());
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
            /*gradients=*/(gradients == c10::nullopt)? nullptr : gradients->data<c10::Half>(),
            /*wp_ends=*/(wp_ends == c10::nullopt)? nullptr : wp_ends->data<int>());
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

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
    m.def("rnnt_loss_sparse(Tensor logits,"
                           "Tensor targets,"
                           "Tensor src_lengths,"
                           "Tensor tgt_lengths,"
                           "int blank,"
                           "float clamp,"
                           "int max_T,"
                           "int max_U,"
                           "Tensor? wp_ends=None,"
                           "int l_buffer=0,"
                           "int r_buffer=0,"
                           "Tensor? valid_ranges=None,"
                           "Tensor? cells_per_sample=None,"
                           "bool fused_log_smax=True,"
                           "bool reuse_logits_for_grads=True) -> (Tensor, Tensor?)",
        &compute_sparse);
}

TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
  m.impl("rnnt_loss", &compute);
  m.impl("rnnt_loss_sparse", &compute_sparse);
}

}  // namespace gpu
}  // namespace rnnt
}  // namespace torchaudio
