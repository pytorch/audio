#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#ifdef USE_CUDA
#include <libtorchaudio/iir_cuda.h>
#endif

namespace {

using torch::stable::Tensor;

template <typename scalar_t>
void host_lfilter_core_loop(
    const Tensor input_signal_windows,
    const Tensor a_coeff_flipped,
    Tensor padded_output_waveform) {
  int64_t n_batch = input_signal_windows.size(0);
  int64_t n_channel = input_signal_windows.size(1);
  int64_t n_samples_input = input_signal_windows.size(2);
  int64_t n_samples_output = padded_output_waveform.size(2);
  int64_t n_order = a_coeff_flipped.size(1);
  scalar_t *output_data;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(padded_output_waveform.get(), (void**)&output_data));
  scalar_t *input_data;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(input_signal_windows.get(), (void**)&input_data));
  scalar_t *a_coeff_flipped_data;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(a_coeff_flipped.get(), (void**)&a_coeff_flipped_data));

  at::parallel_for(0, n_channel * n_batch, 1, [&](int64_t begin, int64_t end) {
    for (auto i = begin; i < end; i++) {
      int64_t offset_input = i * n_samples_input;
      int64_t offset_output = i * n_samples_output;
      int64_t i_channel = i % n_channel;
      for (int64_t i_sample = 0; i_sample < n_samples_input; i_sample++) {
        scalar_t a0 = input_data[offset_input + i_sample];
        for (int64_t i_coeff = 0; i_coeff < n_order; i_coeff++) {
          a0 -= output_data[offset_output + i_sample + i_coeff] *
              a_coeff_flipped_data[i_coeff + i_channel * n_order];
        }
        output_data[offset_output + i_sample + n_order - 1] = a0;
      }
    }
  });
}

void cpu_lfilter_core_loop(
    const Tensor input_signal_windows,
    const Tensor a_coeff_flipped,
    Tensor padded_output_waveform) {
  TORCH_CHECK(
      input_signal_windows.is_cpu() &&
      a_coeff_flipped.is_cpu() &&
      padded_output_waveform.is_cpu());

  TORCH_CHECK(
      input_signal_windows.is_contiguous() && a_coeff_flipped.is_contiguous() &&
      padded_output_waveform.is_contiguous());

  int32_t input_signal_windows_dtype;
  int32_t a_coeff_flipped_dtype;
  int32_t padded_output_waveform_dtype;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(input_signal_windows.get(), &input_signal_windows_dtype));
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(a_coeff_flipped.get(), &a_coeff_flipped_dtype));
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(padded_output_waveform.get(), &padded_output_waveform_dtype));

  TORCH_CHECK(
      (input_signal_windows_dtype == aoti_torch_dtype_float32() ||
       input_signal_windows_dtype == aoti_torch_dtype_float64()) &&
      (a_coeff_flipped_dtype == aoti_torch_dtype_float32() ||
       a_coeff_flipped_dtype == aoti_torch_dtype_float64()) &&
      (padded_output_waveform_dtype == aoti_torch_dtype_float32() ||
       padded_output_waveform_dtype == aoti_torch_dtype_float64()));

  TORCH_CHECK(input_signal_windows.size(0) == padded_output_waveform.size(0));
  TORCH_CHECK(input_signal_windows.size(1) == padded_output_waveform.size(1));

  TORCH_CHECK(
      input_signal_windows.size(2) + a_coeff_flipped.size(1) - 1 ==
      padded_output_waveform.size(2));

  if (input_signal_windows_dtype == aoti_torch_dtype_float32()) {
        host_lfilter_core_loop<float>(
            input_signal_windows, a_coeff_flipped, padded_output_waveform);
  } else if (input_signal_windows_dtype == aoti_torch_dtype_float64()) {
      host_lfilter_core_loop<double>(
          input_signal_windows, a_coeff_flipped, padded_output_waveform);
  }
}

} // namespace

void boxed_cpu_lfilter_core_loop(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor t1(to<AtenTensorHandle>(stack[0]));
  Tensor t2(to<AtenTensorHandle>(stack[1]));
  Tensor t3(to<AtenTensorHandle>(stack[2]));
  cpu_lfilter_core_loop(
      std::move(t1), std::move(t2), std::move(t3));
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "torchaudio::_lfilter_core_loop(Tensor input_signal_windows, Tensor a_coeff_flipped, Tensor(a!) padded_output_waveform) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("torchaudio::_lfilter_core_loop", &boxed_cpu_lfilter_core_loop);
}

// #ifdef USE_CUDA
// STABLE_TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
//   m.impl("torchaudio::_lfilter_core_loop", &cuda_lfilter_core_loop);
// }
// #endif
