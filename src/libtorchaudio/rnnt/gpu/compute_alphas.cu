#include <c10/cuda/CUDAStream.h>
#include <libtorchaudio/rnnt/gpu/gpu_transducer.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/stable/library.h>

namespace torchaudio {
namespace rnnt {
namespace gpu {

using RAIIATH = torch::aot_inductor::RAIIAtenTensorHandle;

RAIIATH compute_alphas(
    const RAIIATH logits,
    const RAIIATH targets,
    const RAIIATH logit_lengths,
    const RAIIATH target_lengths,
    int64_t blank,
    double clamp) {
  Options options;
  int64_t tmp;
  aoti_torch_get_size(logit_lengths.get(), 0, &tmp);
  options.batchSize_ = (int)tmp;
  aoti_torch_get_size(target_lengths.get(), 0, &tmp);
  options.nHypos_ = (int)tmp;
  options.nHypos_ /= options.batchSize_;
  aoti_torch_get_size(logits.get(), 1, &tmp);
  options.maxSrcLen_ = (int)tmp;
  aoti_torch_get_size(logits.get(), 2, &tmp);
  options.maxTgtLen_ = (int)tmp;
  aoti_torch_get_size(logits.get(), 3, &tmp);
  options.numTargets_ = (int)tmp;
  options.blank_ = blank;
  options.clamp_ = clamp;

  int32_t logits_device_type;
  aoti_torch_get_device_type(logits.get(), &logits_device_type);
  AOTI_TORCH_CHECK(logits_device_type == aoti_torch_device_type_cuda());

  int32_t logits_device;
  aoti_torch_get_device_type(logits.get(), &logits_device);
  int32_t logits_device_index;
  aoti_torch_get_device_index(logits.get(), &logits_device_index);
  int32_t logits_dtype;
  aoti_torch_get_dtype(logits.get(), &logits_dtype);

  aoti_torch_get_current_cuda_stream(logits_device_index, &options.stream_);
  cudaSetDevice(logits_device)
  options.device_ = GPU;

  int64_t param_sizes[3] = {options.batchSize_ * options.nHypos_, options.maxSrcLen_, options.maxTgtLen_};
  int64_t param_strides[3] = {options.maxSrcLen_ * options.maxTgtLen_, options.maxTgtLen_, 1};

  AtenTensorHandle alphas;
  aoti_torch_empty_strided(3, param_sizes, param_strides, logits_dtype, logits_device, logits_device_index, &alphas);
  aoti_torch_zero_(alphas);

  AtenTensorHandle int_workspace;
  int64_t sizes[1] = {IntWorkspace::ComputeSizeFromOptions(options)};
  int64_t strides[1] = {1};
  aoti_torch_empty_strided(1, sizes, strides, aoti_torch_dtype_int32(), logits_device, logits_device_index, &int_workspace);

  AtenTensorHandle float_workspace;
  aoti_torch_empty_strided(1, sizes, strides, aoti_torch_dtype_float32(), logits_device, logits_device_index, &float_workspace);

  int64_t float_numel;
  aoti_torch_get_numel(float_workspace, &float_numel);
  void *int_workspace_ptr;
  aoti_torch_get_data_ptr(int_workspace, &int_workspace_ptr);
  void *float_workspace_ptr;
  aoti_torch_get_data_ptr(float_workspace, &float_workspace_ptr);
  int64_t int_numel;
  aoti_torch_get_numel(int_workspace, &int_numel);

  Workspace<float> workspace(
      /*options=*/options,
      /*dtype_data=*/(float*)float_workspace_ptr,
      /*dtype_size=*/float_numel,
      /*int_data=*/(int*)int_workspace_ptr,
      /*int_size=*/int_numel);

  void *logit_ptr;
  aoti_torch_get_data_ptr(logits.get(), &logit_ptr);

  void *target_ptr;
  aoti_torch_get_data_ptr(targets.get(), &target_ptr);

  void *logit_len_ptr;
  aoti_torch_get_data_ptr(logit_lengths.get(), &logit_len_ptr);

  void *target_len_ptr;
  aoti_torch_get_data_ptr(target_lengths.get(), &target_len_ptr);

  void *alpha_ptr;
  aoti_torch_get_data_ptr(alphas, &alpha_ptr);

  // Only support float, this is mainly to enable easy
  // unit-testing
  ComputeAlphas</*DTYPE=*/float, /*CAST_DTYPE=*/float>(
      /*workspace=*/workspace,
      /*logits=*/(float*)logit_ptr,
      /*targets=*/(int*)target_ptr,
      /*logit_lengths=*/(int*)logit_len_ptr,
      /*target_lengths=*/(int*)target_len_ptr,
      /*alphas=*/(float*)alpha_ptr);
  return RAIIATH(alphas);
}

void boxed_compute_alphas(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  RAIIATH t1(to<AtenTensorHandle>(stack[0]));
  RAIIATH t2(to<AtenTensorHandle>(stack[1]));
  RAIIATH t3(to<AtenTensorHandle>(stack[2]));
  RAIIATH t4(to<AtenTensorHandle>(stack[3]));
  int64_t blank = to<int64_t>(stack[4]);
  double clamp = to<double>(stack[5]);
  RAIIATH result = compute_alphas(std::move(t1), std::move(t2), std::move(t3), std::move(t4),
      blank, clamp);
  stack[0] = from(result.release());
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
  m.impl("rnnt_loss_alphas", &boxed_compute_alphas);
}

} // namespace gpu
} // namespace rnnt
} // namespace torchaudio
