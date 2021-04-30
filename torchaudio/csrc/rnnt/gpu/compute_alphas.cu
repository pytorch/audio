#include <THC/THC.h>
#include <torch/script.h>
#include <torchaudio/csrc/rnnt/gpu/gpu_transducer.h>

namespace torchaudio {
namespace rnnt {
namespace gpu {

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

  CHECK_EQ(logits.device().type(), torch::DeviceType::CUDA);
  options.stream_ = at::cuda::getCurrentCUDAStream();
  cudaSetDevice(logits.get_device());
  options.device_ = GPU;

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

torch::Tensor compute_alphas_sparse(
    const torch::Tensor& logits,
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
    const c10::optional<torch::Tensor>& cells_per_sample = c10::nullopt) {
  Options options;
  options.batchSize_ = src_lengths.size(0);
  options.nHypos_ = tgt_lengths.size(0) / src_lengths.size(0);
  options.maxSrcLen_ = max_T;
  options.maxTgtLen_ = max_U;
  options.sparseCells_ = logits.size(0);
  options.numTargets_ = logits.size(1);

  options.blank_ = blank;
  options.clamp_ = clamp;
  options.lBuffer_ = l_buffer;
  options.rBuffer_ = r_buffer;
  options.sparse_ = true;

  CHECK_EQ(logits.device().type(), torch::DeviceType::CUDA);
  options.stream_ = at::cuda::getCurrentCUDAStream();
  cudaSetDevice(logits.get_device());
  options.device_ = GPU;

  torch::Tensor alphas = torch::zeros(
      {logits.size(0)},
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
  ComputeAlphasSparse</*DTYPE=*/float, /*CAST_DTYPE=*/float>(
      /*workspace=*/workspace,
      /*logits=*/logits.data<float>(),
      /*targets=*/targets.data<int>(),
      /*src_lengths=*/src_lengths.data<int>(),
      /*tgt_lengths=*/tgt_lengths.data<int>(),
      /*alphas=*/alphas.data<float>(),
      /*wp_ends=*/(wp_ends == c10::nullopt) ? nullptr : wp_ends->data<int>(),
      (valid_ranges == c10::nullopt) ? nullptr : valid_ranges->data<int>(),
      (cells_per_sample == c10::nullopt) ? nullptr
                                         : cells_per_sample->data<int>());
  return alphas;
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
    m.def("rnnt_loss_alphas_sparse(Tensor logits,"
                                  "Tensor targets,"
                                  "Tensor src_lengths,"
                                  "Tensor tgt_lengths,"
                                  "int blank,"
                                  "float clamp,"
                                  "int max_T,"
                                  "int max_U, Tensor? wp_ends=None,"
                                  "int l_buffer=0,"
                                  "int r_buffer=0,"
                                  "Tensor? valid_ranges=None,"
                                  "Tensor? cells_per_sample=None) -> Tensor",
        &compute_alphas_sparse);
}

TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
  m.impl("rnnt_loss_alphas", &compute_alphas);
  m.impl("rnnt_loss_alphas_sparse", &compute_alphas_sparse);
}

} // namespace gpu
} // namespace rnnt
} // namespace torchaudio
