#include <libtorchaudio/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/TensorAccessor.h>

namespace {
using torch::stable::Tensor;

template <typename scalar_t>
void overdrive_cpu_kernel(
    torchaudio::TensorAccessor<scalar_t, 2> waveform_accessor,
    torchaudio::TensorAccessor<scalar_t, 2> temp_accessor,
    torchaudio::TensorAccessor<scalar_t, 1> last_in_accessor,
    torchaudio::TensorAccessor<scalar_t, 1> last_out_accessor,
    torchaudio::TensorAccessor<scalar_t, 2> output_waveform_accessor) {
  int64_t n_frames = waveform_accessor.size(1);
  int64_t n_channels = waveform_accessor.size(0);

  torch::stable::parallel_for(
      0, n_channels, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i_channel = begin; i_channel < end; ++i_channel) {
          for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {
            last_out_accessor[i_channel] = temp_accessor[i_channel][i_frame] -
                last_in_accessor[i_channel] +
                0.995 * last_out_accessor[i_channel];
            last_in_accessor[i_channel] = temp_accessor[i_channel][i_frame];
            output_waveform_accessor[i_channel][i_frame] =
                waveform_accessor[i_channel][i_frame] * 0.5 +
                last_out_accessor[i_channel] * 0.75;
          }
        }
      });
}

std::tuple<Tensor, Tensor, Tensor> overdrive_core_loop_cpu(
    Tensor waveform,
    Tensor temp,
    Tensor last_in,
    Tensor last_out,
    Tensor output_waveform) {
  THO_DISPATCH_V2(
      waveform.scalar_type(),
      "overdrive_cpu",
      AT_WRAP([&] {
        overdrive_cpu_kernel<scalar_t>(
            torchaudio::accessor<scalar_t, 2>(waveform),
            torchaudio::accessor<scalar_t, 2>(temp),
            torchaudio::accessor<scalar_t, 1>(last_in),
            torchaudio::accessor<scalar_t, 1>(last_out),
            torchaudio::accessor<scalar_t, 2>(output_waveform));
      }),
      AT_FLOATING_TYPES);
  return std::make_tuple(last_in, last_out, output_waveform);
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "_overdrive_core_loop(Tensor waveform,"
      "Tensor temp,"
      "Tensor(a!) last_in,"
      "Tensor(b!) last_out,"
      "Tensor(c!) output_waveform) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("_overdrive_core_loop", TORCH_BOX(&overdrive_core_loop_cpu));
}
