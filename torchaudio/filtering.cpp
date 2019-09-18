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

    // assumes waveform is normalized between 1 and -1
    assert(waveform.dtype == torch::float32);
    assert(a_coeffs.size(0) == b_coeffs.size(0));
    int n_order = a_coeffs.size(0); // n'th order - 1 filter
    assert(n_order > 0);

    int64_t n_channels = waveform.size(0);
    int64_t n_frames = waveform.size(1);

    // Allocate padded input and output waveform, copy in the input waveform
    at::Tensor padded_waveform = torch::zeros({n_channels, n_frames + n_order - 1}, torch::TensorOptions().dtype(torch::kFloat32));
    at::Tensor padded_output_waveform = torch::zeros({n_channels, n_frames + n_order - 1}, torch::TensorOptions().dtype(torch::kFloat32));
    padded_waveform.slice(0, 0, n_channels).slice(1, n_order - 1, n_order + n_frames - 1) = waveform;

    // Create [n_order, n_channels] structure for a and b coefficients
    at::Tensor a_coeffs_filled = at::transpose(at::flip(a_coeffs, {0,}).repeat({n_channels, 1}), 0, 1);
    at::Tensor b_coeffs_filled = at::transpose(at::flip(b_coeffs, {0,}).repeat({n_channels, 1}), 0, 1);

    // few more temporary data structure
    at::Tensor o0 = torch::zeros({n_channels, 1}, torch::TensorOptions().dtype(torch::kFloat32));
    at::Tensor ones = torch::ones({n_channels, n_frames + n_order - 1});
    at::Tensor negones = torch::ones({n_channels, n_frames + n_order -1 }) * -1;
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

    // assumes waveform is normalized between 1 and -1
    assert(waveform.dtype == torch::float32);
    assert(a_coeffs.size(0) == b_coeffs.size(0));
    int n_order = a_coeffs.size(0); // n'th order - 1 filter

    int64_t n_channels = waveform.size(0);
    int64_t n_frames = waveform.size(1);

    // initialize the output tensor
    torch::Tensor output_waveform = torch::zeros({n_channels, n_frames});

    // TBD - Implement for CUDA
    //auto input_accessor = waveform.packed_accessor64<float,2>();
    //auto output_accessor = output_waveform.packed_accessor64<float,2>();    
    //auto a_coeffs_accessor = a_coeffs.packed_accessor64<float,1>();
    //auto b_coeffs_accessor = b_coeffs.packed_accessor64<float,1>();

    auto input_accessor = waveform.accessor<float,2>();
    auto output_accessor = output_waveform.accessor<float,2>();    
    auto a_coeffs_accessor = a_coeffs.accessor<float,1>();
    auto b_coeffs_accessor = b_coeffs.accessor<float,1>();

    for (int64_t i_channel = 0; i_channel < n_channels; ++i_channel) {

      // allocate a temporary data structure of size 2 x (n_order + 1)
      // set to 0 because initial conditions are 0
      float i_s[n_order], o_s[n_order];
      memset( i_s, 0, n_order*sizeof(float) );
      memset( o_s, 0, n_order*sizeof(float) );

      for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {

        // calculate the output at time i_frame by iterating through
        // inputs / outputs at previous time steps and multiplying by coeffs
        i_s[n_order-1] = input_accessor[i_channel][i_frame];
        float o0 = 0;
        for (int i = 0; i < n_order; ++i) {
          o0 += i_s[i] * b_coeffs_accessor[n_order - i - 1];
          if (i != n_order - 1) {
            o0 -= o_s[i] * a_coeffs_accessor[n_order - i - 1];
          }
        }
        o0 = o0 / a_coeffs_accessor[0];

        o_s[n_order-1] = o0;

        // clip and drop into output
        if (o0 > 1) o0 = 1;
        if (o0 < -1) o0 = -1;
        output_accessor[i_channel][i_frame] = o0;

        // shift everything over by one time step
        for (int i = 0; i < (n_order - 1); ++i) {
          i_s[i] = i_s[i+1];
          o_s[i] = o_s[i+1];
        }
      }
    }
    return output_waveform;
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
