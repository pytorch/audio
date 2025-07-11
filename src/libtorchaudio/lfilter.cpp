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

// IIR filter forward and backward functions (no autograd inheritance)
torch::Tensor iir_forward(
    const torch::Tensor& waveform,
    const torch::Tensor& a_coeffs_normalized) {
  auto device = waveform.device();
  auto dtype = waveform.dtype();
  int64_t n_batch = waveform.size(0);
  int64_t n_channel = waveform.size(1);
  int64_t n_sample = waveform.size(2);
  int64_t n_order = a_coeffs_normalized.size(1);
  int64_t n_sample_padded = n_sample + n_order - 1;

  auto a_coeff_flipped = a_coeffs_normalized.flip(1).contiguous();

  auto options = torch::TensorOptions().dtype(dtype).device(device);
  auto padded_output_waveform =
      torch::zeros({n_batch, n_channel, n_sample_padded}, options);

  if (device.is_cpu()) {
    cpu_lfilter_core_loop(waveform, a_coeff_flipped, padded_output_waveform);
  } else if (device.is_cuda()) {
#ifdef USE_CUDA
    cuda_lfilter_core_loop(waveform, a_coeff_flipped, padded_output_waveform);
#else
    lfilter_core_generic_loop(
        waveform, a_coeff_flipped, padded_output_waveform);
#endif
  } else {
    lfilter_core_generic_loop(
        waveform, a_coeff_flipped, padded_output_waveform);
  }

  auto output = padded_output_waveform.index(
      {torch::indexing::Slice(),
       torch::indexing::Slice(),
       torch::indexing::Slice(n_order - 1, torch::indexing::None)});

  return output;
}

std::tuple<torch::Tensor, torch::Tensor> iir_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& waveform,
    const torch::Tensor& a_coeffs_normalized,
    const torch::Tensor& output) {
  int64_t n_channel = waveform.size(1);
  int64_t n_order = a_coeffs_normalized.size(1);

  auto dx = torch::Tensor();
  auto da = torch::Tensor();

  namespace F = torch::nn::functional;

  // Compute tmp using recursive IIR forward (equivalent to DifferentiableIIR::apply)
  auto tmp = iir_forward(grad_output.flip(2).contiguous(), a_coeffs_normalized).flip(2);

  if (waveform.requires_grad()) {
    dx = tmp;
  }

  if (a_coeffs_normalized.requires_grad()) {
    da = -torch::matmul(
              tmp.transpose(0, 1).reshape({n_channel, 1, -1}),
              F::pad(output, F::PadFuncOptions({n_order - 1, 0}))
                  .unfold(2, n_order, 1)
                  .transpose(0, 1)
                  .reshape({n_channel, -1, n_order}))
              .squeeze(1)
              .flip(1);
  }

  return std::make_tuple(dx, da);
}

// FIR filter forward and backward functions (no autograd inheritance)
torch::Tensor fir_forward(
    const torch::Tensor& waveform,
    const torch::Tensor& b_coeffs) {
  int64_t n_order = b_coeffs.size(1);
  int64_t n_channel = b_coeffs.size(0);

  namespace F = torch::nn::functional;
  auto b_coeff_flipped = b_coeffs.flip(1).contiguous();
  auto padded_waveform =
      F::pad(waveform, F::PadFuncOptions({n_order - 1, 0}));

  auto output = F::conv1d(
      padded_waveform,
      b_coeff_flipped.unsqueeze(1),
      F::Conv1dFuncOptions().groups(n_channel));

  return output;
}

std::tuple<torch::Tensor, torch::Tensor> fir_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& waveform,
    const torch::Tensor& b_coeffs) {
  int64_t n_batch = waveform.size(0);
  int64_t n_channel = waveform.size(1);
  int64_t n_order = b_coeffs.size(1);

  auto dx = torch::Tensor();
  auto db = torch::Tensor();

  namespace F = torch::nn::functional;

  // Compute gradient w.r.t. b_coeffs
  if (b_coeffs.requires_grad()) {
    db = F::conv1d(
             F::pad(waveform, F::PadFuncOptions({n_order - 1, 0}))
                 .view({1, n_batch * n_channel, -1}),
             grad_output.view({n_batch * n_channel, 1, -1}),
             F::Conv1dFuncOptions().groups(n_batch * n_channel))
             .view({n_batch, n_channel, -1})
             .sum(0)
             .flip(1);
  }

  // Compute gradient w.r.t. waveform
  if (waveform.requires_grad()) {
    dx = F::conv1d(
        F::pad(grad_output, F::PadFuncOptions({0, n_order - 1})),
        b_coeffs.unsqueeze(1),
        F::Conv1dFuncOptions().groups(n_channel));
  }

  return std::make_tuple(dx, db);
}


} // namespace

// Note: We want to avoid using "catch-all" kernel.
// The following registration should be replaced with CPU specific registration.
TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::_lfilter_core_loop", &cpu_lfilter_core_loop);
}

TORCH_LIBRARY(torchaudio, m) {
  m.def(
      "torchaudio::_differentiable_iir_apply(Tensor waveform, Tensor a_coeffs_normalized) -> Tensor");
  m.def(
      "torchaudio::_fir_forward(Tensor waveform, Tensor b_coeffs) -> Tensor");
  m.def(
      "torchaudio::_fir_backward(Tensor grad_output, Tensor waveform, Tensor b_coeffs) -> (Tensor, Tensor)");
  m.def(
      "torchaudio::_iir_forward(Tensor waveform, Tensor a_coeffs_normalized) -> Tensor");
  m.def(
      "torchaudio::_iir_backward(Tensor grad_output, Tensor waveform, Tensor a_coeffs_normalized, Tensor output) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torchaudio, CompositeImplicitAutograd, m) {
  m.impl("torchaudio::_fir_forward", fir_forward);
  m.impl("torchaudio::_fir_backward", fir_backward);
  m.impl("torchaudio::_iir_forward", iir_forward);
  m.impl("torchaudio::_iir_backward", iir_backward);
}
