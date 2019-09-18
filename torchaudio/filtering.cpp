#include <torch/extension.h>

// TBD - Compile on CUDA
// TBD - Expand to other data types outside float32?

namespace torch {
namespace audio {


at::Tensor lfilter_tensor_matrix(
    at::Tensor const & waveform,
    at::Tensor const & a_coeffs,
    at::Tensor const & b_coeffs
  ) {

    // Assumptions - float32, waveform between -1 and 1
    assert(waveform.dtype() == torch::kFloat32);

    // CPU only for now - Implement for GPU in future
    assert(waveform.device().type() == torch::kCPU);

    // Numerator and denominator coefficients shoudlb e same size
    assert(a_coeffs.size(0) == b_coeffs.size(0));
    int n_order = a_coeffs.size(0); // n'th order - 1 filter
    assert(n_order > 0);

    int64_t n_channels = waveform.size(0);
    int64_t n_frames = waveform.size(1);

    // Device options should mirror input waveform
    auto options = torch::TensorOptions().dtype(waveform.dtype()).device(waveform.device().type());

    // Allocate padded input and output waveform, copy in the input waveform
    at::Tensor padded_waveform = torch::zeros({n_channels, n_frames + n_order - 1}, options);
    at::Tensor padded_output_waveform = torch::zeros({n_channels, n_frames + n_order - 1},  options);
    padded_waveform.slice(0, 0, n_channels).slice(1, n_order - 1, n_order + n_frames - 1) = waveform;

    // Create [n_order, n_channels] structure for a and b coefficients
    at::Tensor a_coeffs_filled = at::transpose(at::flip(a_coeffs, {0,}).repeat({n_channels, 1}), 0, 1);
    at::Tensor b_coeffs_filled = at::transpose(at::flip(b_coeffs, {0,}).repeat({n_channels, 1}), 0, 1);

    // few more temporary data structure
    at::Tensor o0 = torch::zeros({n_channels, 1}, options);
    at::Tensor ones = torch::ones({n_channels, n_frames + n_order - 1}, options);
    at::Tensor negones = torch::ones({n_channels, n_frames + n_order -1 }, options) * -1;
    at::Tensor normalization_a0 = torch::unsqueeze(a_coeffs[0], {0,}).repeat({n_channels, 1});

    for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {

      // calculate the output at time i_frame for all channels
      o0 = torch::zeros({n_channels, 1}, torch::TensorOptions().dtype(torch::kFloat32));

      // time window of input and output, size [n_channels, n_order]
      at::Tensor const & input_window = padded_waveform.slice(0, 0, n_channels).slice(1, i_frame, i_frame + n_order);
      at::Tensor const & output_window = padded_output_waveform.slice(0, 0, n_channels).slice(1, i_frame, i_frame+ n_order);

      // matrix multiply to get [n_channels x n_channels], extract diagonal and unsqueeze to get [n_channels, 1] result for a frame
      at::Tensor inp_result = torch::unsqueeze(torch::diag(torch::mm(input_window, b_coeffs_filled)), 1);
      at::Tensor out_result = torch::unsqueeze(torch::diag(torch::mm(output_window, a_coeffs_filled)), 1);

      o0.add_(inp_result);
      o0.sub_(out_result);

      // normalize by a0
      o0.div_(normalization_a0);

      // Set the output
      padded_output_waveform.slice(0, 0, n_channels).slice(1, i_frame + n_order - 1, i_frame + n_order - 1 + 1) = o0;

    }
    // return clipped, and without initial conditions
    return torch::min(ones, torch::max(negones, padded_output_waveform)).slice(0, 0, n_channels).slice(1, n_order - 1, n_order + n_frames - 1);
  }

  at::Tensor lfilter_element_wise(
    at::Tensor const & waveform,
    at::Tensor const & a_coeffs,
    at::Tensor const & b_coeffs
  ) {

    // Assumptions - float32, waveform between -1 and 1
    assert(waveform.dtype() == torch::kFloat32);

    // CPU only for now - Implement for GPU in future
    assert(waveform.device().type() == torch::kCPU);

    // Numerator and denominator coefficients shoudlb e same size
    assert(a_coeffs.size(0) == b_coeffs.size(0));
    int n_order = a_coeffs.size(0); // n'th order - 1 filter
    assert(n_order > 0);

    int64_t n_channels = waveform.size(0);
    int64_t n_frames = waveform.size(1);

    // Device options should mirror input waveform
    auto options = torch::TensorOptions().dtype(waveform.dtype()).device(waveform.device().type());

    // Allocate padded input and output waveform, copy in the input waveform
    at::Tensor padded_waveform = torch::zeros({n_channels, n_frames + n_order - 1}, options);
    at::Tensor padded_output_waveform = torch::zeros({n_channels, n_frames + n_order - 1},  options);
    padded_waveform.slice(0, 0, n_channels).slice(1, n_order - 1, n_order + n_frames - 1) = waveform;

    // few more temporary data structure
    at::Tensor o0 = torch::zeros({n_channels, 1}, options);
    at::Tensor ones = torch::ones({n_channels, n_frames + n_order - 1}, options);
    at::Tensor negones = torch::ones({n_channels, n_frames + n_order -1 }, options) * -1;
    at::Tensor normalization_a0 = torch::unsqueeze(a_coeffs[0], {0,}).repeat({n_channels, 1});    // Device options should mirror input waveform

    // Create accessors for fast access - https://pytorch.org/cppdocs/notes/tensor_basics.html#efficient-access-to-tensor-elements
    // CPU
    auto input_accessor = padded_waveform.accessor<float,2>();
    auto output_accessor = padded_output_waveform.accessor<float,2>();    
    auto a_coeffs_accessor = a_coeffs.accessor<float,1>();
    auto b_coeffs_accessor = b_coeffs.accessor<float,1>();

    // CUDA - TBD
    //auto input_accessor = waveform.packed_accessor64<float,2>();
    //auto output_accessor = output_waveform.packed_accessor64<float,2>();    
    //auto a_coeffs_accessor = a_coeffs.packed_accessor64<float,1>();
    //auto b_coeffs_accessor = b_coeffs.packed_accessor64<float,1>();
    
    for (int64_t i_channel = 0; i_channel < n_channels; ++i_channel) {

      for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {

        // execute the difference equation
        float o0 = 0;
        for (int i_offset = 0; i_offset < n_order; ++i_offset) {
          o0 += input_accessor[i_channel][i_frame + i_offset] * b_coeffs_accessor[n_order - i_offset - 1];
          o0 -= output_accessor[i_channel][i_frame + i_offset] * a_coeffs_accessor[n_order - i_offset - 1];
        }
        o0 = o0 / a_coeffs_accessor[0];

        // put back into the main data structure
        output_accessor[i_channel][i_frame + n_order - 1] = o0;

      }
    }

    // return clipped, and without initial conditions
    return torch::min(ones, torch::max(negones, padded_output_waveform)).slice(0, 0, n_channels).slice(1, n_order - 1, n_order + n_frames - 1);
  }

}}


PYBIND11_MODULE(_torch_filtering, m) {
  py::options options;
  options.disable_function_signatures();
  m.def(
      "_lfilter_tensor_matrix",
      &torch::audio::lfilter_tensor_matrix,
      "Executes difference equation with tensor");
  m.def(
      "_lfilter_element_wise",
      &torch::audio::lfilter_element_wise,
      "Executes difference equation with tensor");
}
