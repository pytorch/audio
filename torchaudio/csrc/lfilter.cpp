#include <torch/script.h>
#include <torch/torch.h>

namespace {

template <typename scalar_t>
void host_lfilter_core_loop(
    const torch::Tensor& input_signal_windows,
    const torch::Tensor& a_coeff_flipped,
    torch::Tensor& padded_output_waveform) {
  int64_t n_channel = input_signal_windows.size(0);
  int64_t n_samples_input = input_signal_windows.size(1);
  int64_t n_samples_output = padded_output_waveform.size(1);
  int64_t n_order = a_coeff_flipped.size(0);
  scalar_t* output_data = padded_output_waveform.data_ptr<scalar_t>();
  const scalar_t* input_data = input_signal_windows.data_ptr<scalar_t>();
  const scalar_t* a_coeff_flipped_data = a_coeff_flipped.data_ptr<scalar_t>();
  for (int64_t i_channel = 0; i_channel < n_channel; i_channel++) {
    for (int64_t i_sample = 0; i_sample < n_samples_input; i_sample++) {
      int64_t offset_input = i_channel * n_samples_input;
      int64_t offset_output = i_channel * n_samples_output;
      scalar_t a0 = input_data[offset_input + i_sample];
      for (int64_t i_coeff = 0; i_coeff < n_order; i_coeff++) {
        a0 -= output_data[offset_output + i_sample + i_coeff] *
            a_coeff_flipped_data[i_coeff];
      }
      output_data[offset_output + i_sample + n_order - 1] = a0;
    }
  }
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

  TORCH_CHECK(
      input_signal_windows.size(1) + a_coeff_flipped.size(0) - 1 ==
      padded_output_waveform.size(1));

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
  int64_t n_samples_input = input_signal_windows.size(1);
  int64_t n_order = a_coeff_flipped.size(0);
  for (int64_t i_sample = 0; i_sample < n_samples_input; i_sample++) {
    auto windowed_output_signal = padded_output_waveform.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(i_sample, i_sample + n_order)});
    auto o0 = input_signal_windows.index({torch::indexing::Slice(), i_sample})
                  .addmv(windowed_output_signal, a_coeff_flipped, 1, -1);
    padded_output_waveform.index_put_(
        {torch::indexing::Slice(), i_sample + n_order - 1}, o0);
  }
}

torch::Tensor iir_autograd(const torch::Tensor&, const torch::Tensor&);

class DifferentiableIIR : public torch::autograd::Function<DifferentiableIIR> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& waveform,
      const torch::Tensor& a_coeffs) {
    at::AutoNonVariableTypeMode g;
    auto device = waveform.device();
    auto dtype = waveform.dtype();
    int64_t n_channel = waveform.size(0);
    int64_t n_sample = waveform.size(1);
    int64_t n_order = a_coeffs.size(0);
    int64_t n_sample_padded = n_sample + n_order - 1;

    auto a_coeff_flipped = a_coeffs.flip(0).contiguous();
    a_coeff_flipped.div_(a_coeffs[0]);

    auto options = torch::TensorOptions().dtype(dtype).device(device);
    auto padded_output_waveform =
        torch::zeros({n_channel, n_sample_padded}, options);

    if (device.is_cpu()) {
      cpu_lfilter_core_loop(waveform, a_coeff_flipped, padded_output_waveform);
    } else {
      lfilter_core_generic_loop(
          waveform, a_coeff_flipped, padded_output_waveform);
    }

    auto output = padded_output_waveform.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(n_order - 1, torch::indexing::None)});

    ctx->save_for_backward({waveform, a_coeffs, output});
    return output;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto a_coeffs = saved[1];
    auto y = saved[2];

    int64_t n_channel = x.size(0);
    int64_t n_sample = x.size(1);
    int64_t n_order = a_coeffs.size(0);

    auto dx = torch::Tensor();
    auto da = torch::Tensor();
    auto dy = grad_outputs[0];

    namespace F = torch::nn::functional;

    if (a_coeffs.requires_grad()) {
      auto dyda = F::pad(
          iir_autograd(-y, a_coeffs), F::PadFuncOptions({n_order - 1, 0}));

      da = F::conv1d(
               dyda.unsqueeze(0),
               dy.unsqueeze(1),
               F::Conv1dFuncOptions().groups(n_channel))
               .sum(1)
               .squeeze(0)
               .flip(0);
      // use .item() to detach from graph
      da.div_(a_coeffs[0].item());
    }

    if (x.requires_grad()) {
      dx = iir_autograd(dy.flip(1), a_coeffs).flip(1);
    }

    return {dx, da};
  }
};

torch::Tensor iir_autograd(
    const torch::Tensor& waveform,
    const torch::Tensor& a_coeffs) {
  return DifferentiableIIR::apply(waveform, a_coeffs);
}

torch::Tensor lfilter_core(
    const torch::Tensor& waveform,
    const torch::Tensor& a_coeffs,
    const torch::Tensor& b_coeffs) {
  TORCH_CHECK(waveform.device() == a_coeffs.device());
  TORCH_CHECK(b_coeffs.device() == a_coeffs.device());
  TORCH_CHECK(a_coeffs.size(0) == b_coeffs.size(0));

  TORCH_INTERNAL_ASSERT(waveform.sizes().size() == 2);

  int64_t n_order = b_coeffs.size(0);

  TORCH_INTERNAL_ASSERT(n_order > 0);

  namespace F = torch::nn::functional;

  auto padded_waveform = F::pad(waveform, F::PadFuncOptions({n_order - 1, 0}));
  auto b_coeff_flipped = b_coeffs.flip(0).contiguous();

  // use .item() to detach a[0] from graph
  b_coeff_flipped.div_(a_coeffs[0].item());

  auto filtered_waveform =
      F::conv1d(
          padded_waveform.unsqueeze(1), b_coeff_flipped.view({1, 1, n_order}))
          .squeeze(1);

  auto output = iir_autograd(filtered_waveform, a_coeffs);
  return output;
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
}

TORCH_LIBRARY_IMPL(torchaudio, CompositeImplicitAutograd, m) {
  m.impl("torchaudio::_lfilter", lfilter_core);
}
