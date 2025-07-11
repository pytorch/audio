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

class DifferentiableIIR : public torch::autograd::Function<DifferentiableIIR> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
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

    ctx->save_for_backward({waveform, a_coeffs_normalized, output});
    return output;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto a_coeffs_normalized = saved[1];
    auto y = saved[2];

    int64_t n_channel = x.size(1);
    int64_t n_order = a_coeffs_normalized.size(1);

    auto dx = torch::Tensor();
    auto da = torch::Tensor();
    auto dy = grad_outputs[0];

    namespace F = torch::nn::functional;

    auto tmp =
        DifferentiableIIR::apply(dy.flip(2).contiguous(), a_coeffs_normalized)
            .flip(2);

    if (x.requires_grad()) {
      dx = tmp;
    }

    if (a_coeffs_normalized.requires_grad()) {
      da = -torch::matmul(
                tmp.transpose(0, 1).reshape({n_channel, 1, -1}),
                F::pad(y, F::PadFuncOptions({n_order - 1, 0}))
                    .unfold(2, n_order, 1)
                    .transpose(0, 1)
                    .reshape({n_channel, -1, n_order}))
                .squeeze(1)
                .flip(1);
    }
    return {dx, da};
  }
};

class DifferentiableFIR : public torch::autograd::Function<DifferentiableFIR> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
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

    ctx->save_for_backward({waveform, b_coeffs, output});
    return output;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto b_coeffs = saved[1];
    auto y = saved[2];

    int64_t n_batch = x.size(0);
    int64_t n_channel = x.size(1);
    int64_t n_order = b_coeffs.size(1);

    auto dx = torch::Tensor();
    auto db = torch::Tensor();
    auto dy = grad_outputs[0];

    namespace F = torch::nn::functional;

    if (b_coeffs.requires_grad()) {
      db = F::conv1d(
               F::pad(x, F::PadFuncOptions({n_order - 1, 0}))
                   .view({1, n_batch * n_channel, -1}),
               dy.view({n_batch * n_channel, 1, -1}),
               F::Conv1dFuncOptions().groups(n_batch * n_channel))
               .view({n_batch, n_channel, -1})
               .sum(0)
               .flip(1);
    }

    if (x.requires_grad()) {
      dx = F::conv1d(
          F::pad(dy, F::PadFuncOptions({0, n_order - 1})),
          b_coeffs.unsqueeze(1),
          F::Conv1dFuncOptions().groups(n_channel));
    }

    return {dx, db};
  }
};

torch::Tensor differentiable_fir_apply(
    const torch::Tensor& waveform,
    const torch::Tensor& b_coeffs) {
  return DifferentiableFIR::apply(waveform, b_coeffs);
}

torch::Tensor differentiable_iir_apply(
    const torch::Tensor& waveform,
    const torch::Tensor& a_coeffs_normalized) {
  return DifferentiableIIR::apply(waveform, a_coeffs_normalized);
}

} // namespace

// Note: We want to avoid using "catch-all" kernel.
// The following registration should be replaced with CPU specific registration.
TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::_lfilter_core_loop", &cpu_lfilter_core_loop);
}

TORCH_LIBRARY(torchaudio, m) {
  m.def(
      "torchaudio::_lfilter(Tensor waveform, Tensor a_coeffs, Tensor b_coeffs) -> Tensor");
  m.def(
      "torchaudio::_differentiable_fir_apply(Tensor waveform, Tensor b_coeffs) -> Tensor");
  m.def(
      "torchaudio::_differentiable_iir_apply(Tensor waveform, Tensor a_coeffs_normalized) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchaudio, CompositeImplicitAutograd, m) {
  m.impl("torchaudio::_differentiable_fir_apply", differentiable_fir_apply);
  m.impl("torchaudio::_differentiable_iir_apply", differentiable_iir_apply);
}
