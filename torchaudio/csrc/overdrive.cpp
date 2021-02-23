#include <torch/script.h>

namespace {

template <typename scalar_t>
void overdrive_cpu_kernel(
    at::TensorAccessor<scalar_t, 2> waveform_accessor,
    at::TensorAccessor<scalar_t, 2> temp_accessor,
    at::TensorAccessor<scalar_t, 1> last_in_accessor,
    at::TensorAccessor<scalar_t, 1> last_out_accessor,
    at::TensorAccessor<scalar_t, 2> output_waveform_accessor) {
  int64_t n_frames = waveform_accessor.size(1);
  int64_t n_channels = waveform_accessor.size(0);

  for (int64_t i_channel = 0; i_channel < n_channels; ++i_channel) {
    for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {
      last_out_accessor[i_channel] = temp_accessor[i_channel][i_frame] -
          last_in_accessor[i_channel] + 0.995 * last_out_accessor[i_channel];
      last_in_accessor[i_channel] = temp_accessor[i_channel][i_frame];
      output_waveform_accessor[i_channel][i_frame] =
          waveform_accessor[i_channel][i_frame] * 0.5 +
          last_out_accessor[i_channel] * 0.75;
    }
  }
}

void overdrive_core_loop_cpu(
    at::Tensor& waveform,
    at::Tensor& temp,
    at::Tensor& last_in,
    at::Tensor& last_out,
    at::Tensor& output_waveform) {
  AT_DISPATCH_FLOATING_TYPES(waveform.scalar_type(), "overdrive_cpu", ([&] {
                               overdrive_cpu_kernel<scalar_t>(
                                   waveform.accessor<scalar_t, 2>(),
                                   temp.accessor<scalar_t, 2>(),
                                   last_in.accessor<scalar_t, 1>(),
                                   last_out.accessor<scalar_t, 1>(),
                                   output_waveform.accessor<scalar_t, 2>());
                             }));
}

} // namespace

// Note: We want to avoid using "catch-all" kernel.
// The following registration should be replaced with CPU specific registration.
TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::_overdrive_core_loop", &overdrive_core_loop_cpu);
}
