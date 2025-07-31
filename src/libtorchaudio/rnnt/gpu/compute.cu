#include <c10/cuda/CUDAStream.h>
#include <libtorchaudio/rnnt/gpu/gpu_transducer.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/stable/library.h>

namespace torchaudio {
namespace rnnt {
namespace gpu {

using torch::stable::Tensor;

// Entry point into RNNT Loss
std::tuple<Tensor, Tensor> compute(
    const Tensor logits,
    const Tensor targets,
    const Tensor logit_lengths,
    const Tensor target_lengths,
    int64_t blank,
    double clamp,
    bool fused_log_softmax = true) {

    int32_t logits_device;
    aoti_torch_get_device_type(logits.get(), &logits_device);
    int32_t targets_device;
    aoti_torch_get_device_type(targets.get(), &targets_device);
    int32_t logit_lengths_device;
    aoti_torch_get_device_type(logit_lengths.get(), &logit_lengths_device);
    int32_t target_lengths_device;
    aoti_torch_get_device_type(target_lengths.get(), &target_lengths_device);

    AOTI_TORCH_CHECK(logits_device == targets_device);
    AOTI_TORCH_CHECK(logits_device == logit_lengths_device);
    AOTI_TORCH_CHECK(logits_device == target_lengths_device);

    int32_t logits_dtype;
    aoti_torch_get_dtype(logits.get(), &logits_dtype);
    AOTI_TORCH_CHECK(logits_dtype == aoti_torch_dtype_float32() ||
      logits_dtype == aoti_torch_dtype_float16());

    int32_t targets_dtype;
    aoti_torch_get_dtype(targets.get(), &targets_dtype);
    AOTI_TORCH_CHECK(targets_dtype == aoti_torch_dtype_int32() ||
      logits_dtype == aoti_torch_dtype_float16());

    int32_t logit_lengths_dtype;
    aoti_torch_get_dtype(logit_lengths.get(), &logit_lengths_dtype);
    AOTI_TORCH_CHECK(logit_lengths_dtype == aoti_torch_dtype_int32() ||
      logit_lengths_dtype == aoti_torch_dtype_float16());

    int32_t target_lengths_dtype;
    aoti_torch_get_dtype(target_lengths.get(), &target_lengths_dtype);
    AOTI_TORCH_CHECK(target_lengths_dtype == aoti_torch_dtype_int32() ||
      target_lengths_dtype == aoti_torch_dtype_float16());

    bool bool_tmp;
    aoti_torch_is_contiguous(logits.get(), &bool_tmp);
    AOTI_TORCH_CHECK(bool_tmp);
    aoti_torch_is_contiguous(targets.get(), &bool_tmp);
    AOTI_TORCH_CHECK(bool_tmp);
    aoti_torch_is_contiguous(logit_lengths.get(), &bool_tmp);
    AOTI_TORCH_CHECK(bool_tmp);
    aoti_torch_is_contiguous(target_lengths.get(), &bool_tmp);

    int64_t int_tmp;
    aoti_torch_get_dim(logits.get(), &int_tmp);
    AOTI_TORCH_CHECK(int_tmp == 4);
    aoti_torch_get_dim(targets.get(), &int_tmp);
    AOTI_TORCH_CHECK(int_tmp == 2);
    aoti_torch_get_dim(logit_lengths.get(), &int_tmp);
    AOTI_TORCH_CHECK(int_tmp == 1);
    aoti_torch_get_dim(target_lengths.get(), &int_tmp);
    AOTI_TORCH_CHECK(int_tmp == 1);

    int64_t logit_lengths_size;
    aoti_torch_get_size(logit_lengths.get(), 0, &logit_lengths_size);
    int64_t logits_size;
    aoti_torch_get_size(logits.get(), 0, &logits_size);
    AOTI_TORCH_CHECK(logit_lengths_size == logits_size);
    int64_t target_lengths_size;
    aoti_torch_get_size(target_lengths.get(), 0, &target_lengths_size);
    AOTI_TORCH_CHECK(target_lengths_size == logits_size);
    int64_t targets_size;
    aoti_torch_get_size(targets.get(), 0, &targets_size);
    AOTI_TORCH_CHECK(targets_size == logits_size);

    // TORCH_CHECK(
    //     blank >= 0 && blank < logits.size(-1),
    //     "blank must be within [0, logits.shape[-1])");

    // TORCH_CHECK(
    //     logits.size(1) == at::max(logit_lengths).item().toInt(),
    //     "input length mismatch");
    // TORCH_CHECK(
    //     logits.size(2) == at::max(target_lengths).item().toInt() + 1,
    //     "output length mismatch");
    // TORCH_CHECK(
    //     targets.size(1) == at::max(target_lengths).item().toInt(),
    //     "target length mismatch");

    Options options;
    options.batchSize_ = (int)logit_lengths_size;
    options.nHypos_ = (int)target_lengths_size;
    options.nHypos_ /= options.batchSize_;
    aoti_torch_get_size(logits.get(), 1, &int_tmp);
    options.maxSrcLen_ = (int)int_tmp;
    aoti_torch_get_size(logits.get(), 2, &int_tmp);
    options.maxTgtLen_ = (int)int_tmp;
    aoti_torch_get_size(logits.get(), 3, &int_tmp);
    options.numTargets_ = (int)int_tmp;
    options.blank_ = blank;
    options.clamp_ = clamp;
    options.fusedLogSmax_ = fused_log_softmax;

  int32_t logits_device_index;
  aoti_torch_get_device_index(logits.get(), &logits_device_index);

  TORCH_CHECK_EQ(logits_device, aoti_torch_device_type_cuda());
  aoti_torch_get_current_cuda_stream(logits_device_index, (void**)&options.stream_);
  cudaSetDevice(logits_device);
  options.device_ = GPU;

  int64_t cost_sizes[1] = {options.batchSize_ * options.nHypos_};
  int64_t stride1[1] = {1};
  AtenTensorHandle costs;
  aoti_torch_empty_strided(1, cost_sizes, stride1, logits_dtype, logits_device, logits_device_index, &costs);

  AtenTensorHandle gradients;
  aoti_torch_clone(logits.get(), &gradients);
  aoti_torch_zero_(gradients);

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

  void *costs_ptr;
  aoti_torch_get_data_ptr(costs, &costs_ptr);

  void *grads_ptr;
  aoti_torch_get_data_ptr(gradients, &grads_ptr);

  if (logits_dtype == aoti_torch_dtype_float32()) {
      Compute</*DTYPE=*/float, /*CAST_DTYPE=*/float>(
          /*workspace=*/workspace,
          /*logits=*/(float*)logit_ptr,
          /*targets=*/(int*)target_ptr,
          /*logit_lengths=*/(int*)logit_len_ptr,
          /*target_lengths=*/(int*)target_len_ptr,
          /*costs=*/(float*)costs_ptr,
          /*gradients=*/(float*)grads_ptr);
    } else {
      Compute</*DTYPE=*/c10::Half, /*CAST_DTYPE=*/float>(
          /*workspace=*/workspace,
          /*logits=*/(c10::Half*)logit_ptr,
          /*targets=*/(int*)target_ptr,
          /*logit_lengths=*/(int*)logit_len_ptr,
          /*target_lengths=*/(int*)target_len_ptr,
          /*costs=*/(c10::Half*)costs_ptr,
          /*gradients=*/(c10::Half*)grads_ptr);
    }

  return std::make_tuple(Tensor(costs), Tensor(gradients));
}

void boxed_compute(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor t1(to<AtenTensorHandle>(stack[0]));
  Tensor t2(to<AtenTensorHandle>(stack[1]));
  Tensor t3(to<AtenTensorHandle>(stack[2]));
  Tensor t4(to<AtenTensorHandle>(stack[3]));
  int64_t blank = to<int64_t>(stack[4]);
  double clamp = to<double>(stack[5]);
  bool fused_log_softmax = to<bool>(stack[6]);
  auto result = compute(
      std::move(t1), std::move(t2), std::move(t3), std::move(t4),
      blank, clamp, fused_log_softmax);
  stack[0] = from(std::get<0>(result));
  stack[1] = from(std::get<1>(result));
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
  m.impl("rnnt_loss", &boxed_compute);
}

} // namespace gpu
} // namespace rnnt
} // namespace torchaudio
