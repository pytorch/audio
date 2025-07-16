#include <torch/script.h>
#include <torch/torch.h>

#ifdef USE_CUDA
#include <libtorchaudio/iir_cuda.h>
#endif

namespace {

template <typename scalar_t>
void host_lfilter_core_loop(
    const torch::Tensor& input_signal_windows,
    const torch::Tensor& a_coeff_flipped,
    torch::Tensor& padded_output_waveform) {
  int64_t n_batch = input_signal_windows.size(0);
  int64_t n_channel = input_signal_windows.size(1);
  int64_t n_samples_input = input_signal_windows.size(2);
  int64_t n_samples_output = padded_output_waveform.size(2);
  int64_t n_order = a_coeff_flipped.size(1);
  scalar_t* output_data = padded_output_waveform.data_ptr<scalar_t>();
  const scalar_t* input_data = input_signal_windows.data_ptr<scalar_t>();
  const scalar_t* a_coeff_flipped_data = a_coeff_flipped.data_ptr<scalar_t>();

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
    const torch::Tensor& input_signal_windows,
    const torch::Tensor& a_coeff_flipped,
    torch::Tensor& padded_output_waveform) {
  TORCH_CHECK(
      input_signal_windows.device().is_cpu() &&
      a_coeff_flipped.device().is_cpu() &&
      padded_output_waveform.device().is_cpu());

  TORCH_CHECK(
      input_signal_windows.is_contiguous() && a_coeff_flipped.is_contiguous() &&
      padded_output_waveform.is_contiguous());

  TORCH_CHECK(
      (input_signal_windows.dtype() == torch::kFloat32 ||
       input_signal_windows.dtype() == torch::kFloat64) &&
      (a_coeff_flipped.dtype() == torch::kFloat32 ||
       a_coeff_flipped.dtype() == torch::kFloat64) &&
      (padded_output_waveform.dtype() == torch::kFloat32 ||
       padded_output_waveform.dtype() == torch::kFloat64));

  TORCH_CHECK(input_signal_windows.size(0) == padded_output_waveform.size(0));
  TORCH_CHECK(input_signal_windows.size(1) == padded_output_waveform.size(1));

  TORCH_CHECK(
      input_signal_windows.size(2) + a_coeff_flipped.size(1) - 1 ==
      padded_output_waveform.size(2));

  AT_DISPATCH_FLOATING_TYPES(
      input_signal_windows.scalar_type(), "lfilter_core_loop", [&] {
        host_lfilter_core_loop<scalar_t>(
            input_signal_windows, a_coeff_flipped, padded_output_waveform);
      });
}

void lfilter_core_generic_loop(
    const torch::Tensor& input_signal_windows,
    const torch::Tensor& a_coeff_flipped,
    torch::Tensor& padded_output_waveform) {
  int64_t n_samples_input = input_signal_windows.size(2);
  int64_t n_order = a_coeff_flipped.size(1);
  auto coeff = a_coeff_flipped.unsqueeze(2);
  for (int64_t i_sample = 0; i_sample < n_samples_input; i_sample++) {
    auto windowed_output_signal =
        padded_output_waveform
            .index(
                {torch::indexing::Slice(),
                 torch::indexing::Slice(),
                 torch::indexing::Slice(i_sample, i_sample + n_order)})
            .transpose(0, 1);
    auto o0 =
        input_signal_windows.index(
            {torch::indexing::Slice(), torch::indexing::Slice(), i_sample}) -
        at::matmul(windowed_output_signal, coeff).squeeze(2).transpose(0, 1);
    padded_output_waveform.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(),
         i_sample + n_order - 1},
        o0);
  }
}

} // namespace

TORCH_LIBRARY(torchaudio, m) {
  m.def(
      "torchaudio::_lfilter_core_loop(Tensor input_signal_windows, Tensor a_coeff_flipped, Tensor(a!) padded_output_waveform) -> ()");
}

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("torchaudio::_lfilter_core_loop", &cpu_lfilter_core_loop);
}

#ifdef USE_CUDA
TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
  m.impl("torchaudio::_lfilter_core_loop", &cuda_lfilter_core_loop);
}
#endif

TORCH_LIBRARY_IMPL(torchaudio, CompositeExplicitAutograd, m) {
  m.impl("torchaudio::_lfilter_core_loop", &lfilter_core_generic_loop);
}
