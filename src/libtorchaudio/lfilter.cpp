#include <libtorchaudio/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>

#ifdef USE_CUDA
#include <libtorchaudio/iir_cuda.h>
#endif

namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

template <typename scalar_t>
void host_lfilter_core_loop(
    const Tensor& input_signal_windows,
    const Tensor& a_coeff_flipped,
    Tensor& padded_output_waveform) {
  int64_t n_batch = input_signal_windows.size(0);
  int64_t n_channel = input_signal_windows.size(1);
  int64_t n_samples_input = input_signal_windows.size(2);
  int64_t n_samples_output = padded_output_waveform.size(2);
  int64_t n_order = a_coeff_flipped.size(1);
  scalar_t* output_data =
      reinterpret_cast<scalar_t*>(padded_output_waveform.data_ptr());
  const scalar_t* input_data =
      reinterpret_cast<scalar_t*>(input_signal_windows.data_ptr());
  const scalar_t* a_coeff_flipped_data =
      reinterpret_cast<scalar_t*>(a_coeff_flipped.data_ptr());

  torch::stable::parallel_for(
      0, n_channel * n_batch, 1, [&](int64_t begin, int64_t end) {
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

Tensor cpu_lfilter_core_loop(
    Tensor input_signal_windows,
    Tensor a_coeff_flipped,
    Tensor padded_output_waveform) {
  STD_TORCH_CHECK(
      input_signal_windows.is_cpu() && a_coeff_flipped.is_cpu() &&
      padded_output_waveform.is_cpu());

  STD_TORCH_CHECK(
      input_signal_windows.is_contiguous() && a_coeff_flipped.is_contiguous() &&
      padded_output_waveform.is_contiguous());

  STD_TORCH_CHECK(
      (input_signal_windows.scalar_type() == ScalarType::Float ||
       input_signal_windows.scalar_type() == ScalarType::Double) &&
      (a_coeff_flipped.scalar_type() == ScalarType::Float ||
       a_coeff_flipped.scalar_type() == ScalarType::Double) &&
      (padded_output_waveform.scalar_type() == ScalarType::Float ||
       padded_output_waveform.scalar_type() == ScalarType::Double));

  STD_TORCH_CHECK(
      input_signal_windows.size(0) == padded_output_waveform.size(0));
  STD_TORCH_CHECK(
      input_signal_windows.size(1) == padded_output_waveform.size(1));

  STD_TORCH_CHECK(
      input_signal_windows.size(2) + a_coeff_flipped.size(1) - 1 ==
      padded_output_waveform.size(2));

  THO_DISPATCH_V2(
      input_signal_windows.scalar_type(),
      "lfilter_core_loop",
      [&] {
        host_lfilter_core_loop<scalar_t>(
            input_signal_windows, a_coeff_flipped, padded_output_waveform);
      },
      AT_FLOATING_TYPES);
  return padded_output_waveform;
}

Tensor lfilter_core_generic_loop(
    Tensor input_signal_windows,
    Tensor a_coeff_flipped,
    Tensor padded_output_waveform) {
  int64_t n_samples_input = input_signal_windows.size(2);
  int64_t n_order = a_coeff_flipped.size(1);
  auto coeff = torch::stable::unsqueeze(a_coeff_flipped, 2);
  for (int64_t i_sample = 0; i_sample < n_samples_input; i_sample++) {
    auto windowed_output_signal = torch::stable::transpose(
        torch::stable::narrow(padded_output_waveform, 2, i_sample, n_order),
        0,
        1);
    auto o0 = torch::stable::subtract(
        torch::stable::select(input_signal_windows, 2, i_sample),
        torch::stable::transpose(
            torch::stable::squeeze(
                torch::stable::matmul(windowed_output_signal, coeff), 2),
            0,
            1));
    auto s = torch::stable::select(
        padded_output_waveform, 2, i_sample + n_order - 1);
    torch::stable::copy_(s, o0);
  }
  return padded_output_waveform;
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "_lfilter_core_loop("
      "Tensor input_signal_windows,"
      "Tensor a_coeff_flipped,"
      "Tensor(a!) padded_output_waveform) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("_lfilter_core_loop", TORCH_BOX(&cpu_lfilter_core_loop));
}

#ifdef USE_CUDA
STABLE_TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
  m.impl("_lfilter_core_loop", TORCH_BOX(&cuda_lfilter_core_loop));
}
#endif

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CompositeExplicitAutograd, m) {
  m.impl("_lfilter_core_loop", TORCH_BOX(&lfilter_core_generic_loop));
}
