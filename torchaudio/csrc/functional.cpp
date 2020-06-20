#include <torchaudio/csrc/functional.h>

using namespace torch::indexing;

namespace torchaudio {
namespace functional {

torch::Tensor lfilter(
  torch::Tensor waveform,
  torch::Tensor a_coeffs,
  torch::Tensor b_coeffs,
  const bool clamp) {

  auto option = torch::TensorOptions()
    .dtype(waveform.dtype())
    .device(waveform.device());

  const auto input_shape = waveform.sizes();
  const auto num_time = input_shape[input_shape.size() -1];
  auto tensor = waveform.reshape({-1, num_time});

  const auto tensor_shape = tensor.sizes();
  const auto num_channels = tensor_shape[0];
  const auto num_frames = tensor_shape[1];
  const auto n_order = a_coeffs.size(0);
  const auto num_sample_padded = num_frames + n_order - 1;

  auto padded_waveform = at::constant_pad_nd(tensor, {n_order-1, 0}, 0);
  auto padded_output_waveform = torch::zeros({num_channels, num_sample_padded}, option);

  auto a_coeffs_flipped = a_coeffs.flip(0);
  auto b_coeffs_flipped = b_coeffs.flip(0);

  const auto window_idxs = [&]() {
    auto idx = (torch::arange(num_frames, option).unsqueeze(0) +
                torch::arange(n_order, option).unsqueeze(1));
    idx = idx.repeat({num_channels, 1, 1});
    idx += torch::arange(num_channels, option).unsqueeze(-1).unsqueeze(-1) * num_sample_padded;
    return idx.to(torch::kInt64);
  }();

  auto input_signal_windows = [&]() {
    auto t = torch::matmul(b_coeffs_flipped, at::take(padded_waveform, window_idxs));
    t.true_divide_(a_coeffs[0]);
    return t.t();
  }();

  a_coeffs_flipped.true_divide_(a_coeffs[0]);
  for (int64_t i = 0; i < input_signal_windows.sizes()[0]; ++i) {
    auto windowed_output_signal = padded_output_waveform.index({Slice(), Slice(i, i+n_order)});
    auto o0 = input_signal_windows.index({i});
    at::addmv_(o0, windowed_output_signal, a_coeffs_flipped, 1, -1);
    padded_output_waveform.index_put_({Slice(), i + n_order - 1}, o0);
  }

  tensor = padded_output_waveform.index({Slice(), Slice(n_order - 1)});

  if (clamp) {
    tensor.clamp_(-1., 1.);
  }

  return tensor.reshape(input_shape);
};

}
}
