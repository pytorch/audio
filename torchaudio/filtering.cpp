#include <torch/extension.h>
// TBD - for CUDA support #include <ATen/cuda/CUDAContext.h>
// TBD - Compile on CUDA

namespace torch {
namespace audio {


void _lfilter_tensor_matrix(
    at::Tensor const & padded_waveform,
    at::Tensor & padded_output_waveform,
    at::Tensor const & a_coeffs_filled,
    at::Tensor const & b_coeffs_filled,
    at::Tensor & o0,
    at::Tensor const & normalization_a0
  ) {
    int64_t n_order = a_coeffs_filled.size(0);
    int64_t n_frames = padded_waveform.size(1) - n_order + 1;
    int64_t n_channels = padded_waveform.size(0);

    for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {
      // reset all o0
      o0.fill_(0.0);

      // time window of input and output, size [n_channels, n_order]
      at::Tensor const & input_window =
          padded_waveform.slice(0, 0, n_channels)
                         .slice(1, i_frame, i_frame + n_order);
      at::Tensor const & output_window =
          padded_output_waveform.slice(0, 0, n_channels)
                                .slice(1, i_frame, i_frame+ n_order);

      // matrix multiply to get [n_channels x n_channels],
      // extract diagonal and unsqueeze to get [n_channels, 1] result
      at::Tensor inp_result =
          torch::unsqueeze(torch::diag(torch::mm(input_window,
                                                 b_coeffs_filled)), 1);
      at::Tensor out_result =
          torch::unsqueeze(torch::diag(torch::mm(output_window,
                                                 a_coeffs_filled)), 1);

      o0.add_(inp_result);
      o0.sub_(out_result);

      // normalize by a0
      o0.div_(normalization_a0);

      // Set the output
      padded_output_waveform.slice(0, 0, n_channels)
                            .slice(1,
                                   i_frame + n_order - 1,
                                   i_frame + n_order - 1 + 1) = o0;
    }
  }

  template <typename T>
  void _lfilter_element_wise(
    at::Tensor const & padded_waveform,
    at::Tensor & padded_output_waveform,
    at::Tensor const & a_coeffs,
    at::Tensor const & b_coeffs
  ) {
    int64_t n_order = a_coeffs.size(0);
    int64_t n_frames = padded_waveform.size(1) - n_order + 1;
    int64_t n_channels = padded_waveform.size(0);

    // Create CPU accessors for fast access
    // https://pytorch.org/cppdocs/notes/tensor_basics.html
    auto input_accessor = padded_waveform.accessor<T, 2>();
    auto output_accessor = padded_output_waveform.accessor<T, 2>();
    auto a_coeffs_accessor = a_coeffs.accessor<T, 1>();
    auto b_coeffs_accessor = b_coeffs.accessor<T, 1>();
    T o0;

    for (int64_t i_channel = 0; i_channel < n_channels; ++i_channel) {
      for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {
        // execute the difference equation
        o0 = 0;
        for (int i_offset = 0; i_offset < n_order; ++i_offset) {
          o0 += input_accessor[i_channel][i_frame + i_offset] *
                b_coeffs_accessor[n_order - i_offset - 1];
          o0 -= output_accessor[i_channel][i_frame + i_offset] *
                a_coeffs_accessor[n_order - i_offset - 1];
        }
        o0 = o0 / a_coeffs_accessor[0];

        // put back into the main data structure
        output_accessor[i_channel][i_frame + n_order - 1] = o0;
      }
    }
  }

}  // namespace audio
}  // namespace torch


PYBIND11_MODULE(_torch_filtering, m) {
  py::options options;
  options.disable_function_signatures();
  m.def(
      "_lfilter_tensor_matrix",
      &torch::audio::_lfilter_tensor_matrix,
      "Executes difference equation with tensor");
  m.def(
      "_lfilter_element_wise_float",
      &torch::audio::_lfilter_element_wise<float>,
      "Executes difference equation with tensor");
  m.def(
      "_lfilter_element_wise_double",
      &torch::audio::_lfilter_element_wise<double>,
      "Executes difference equation with tensor");
}
