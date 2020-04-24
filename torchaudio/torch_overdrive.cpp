#include <torch/extension.h>
// TBD - for CUDA support #include <ATen/cuda/CUDAContext.h>
// TBD - Compile on CUDA

namespace torch {
namespace audio {


  template <typename T>
  void _overdrive_float(
    at::Tensor & waveform,
    at::Tensor & temp,
    at::Tensor & last_in,
    at::Tensor & last_out,
    at::Tensor & output_waveform
  ) {
    int64_t n_frames = waveform.size(1);
    int64_t n_channels = waveform.size(0);
    // Create CPU accessors for fast access
    // https://pytorch.org/cppdocs/notes/tensor_basics.html
    auto waveform_accessor = waveform.accessor<T, 2>();
    auto temp_accessor = temp.accessor<T, 2>();
    auto last_in_accessor = last_in.accessor<T, 1>();
    auto last_out_accessor = last_out.accessor<T, 1>();
    auto output_waveform_accessor = output_waveform.accessor<T, 2>();

    for (int64_t i_channel = 0; i_channel < n_channels; ++i_channel) {

        for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {
        last_out_accessor[i_channel] = temp_accessor[i_channel][i_frame] - last_in_accessor[i_channel] + 0.995 * last_out_accessor[i_channel];
        last_in_accessor[i_channel] = temp_accessor[i_channel][i_frame];
        output_waveform_accessor[i_channel][i_frame]= waveform_accessor[i_channel][i_frame] * 0.5 + last_out_accessor[i_channel] * 0.75;
        }
    }
    /*
    for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {

      last_out = temp.slice(1,i_frame,i_frame+1) - last_in + 0.995 * last_out;
      last_in = temp.slice(1,i_frame,i_frame+1);
      output_waveform.slice(1,i_frame,i_frame+1) = waveform.slice(1,i_frame,i_frame+1) * 0.5 + last_out * 0.75;

      }
    */

  }

}  // namespace audio
}  // namespace torch


PYBIND11_MODULE(_torch_overdrive, m) {
  m.def(
      "_overdrive_float",
      &torch::audio::_overdrive_float<float>,
      "Executes difference equation with tensor");
}
