#include <torch/extension.h>
#ifdef WITH_CUDA
#include <cuda.h>
#endif

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

at::Tensor lfilter_tensor(
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

    // Create [n_channels, n_order] structure for a and b coefficients
    at::Tensor a_coeffs_filled = a_coeffs.repeat({n_channels, 1});
    at::Tensor b_coeffs_filled = b_coeffs.repeat({n_channels, 1});

    // few more temporary data structure
    at::Tensor o0 = torch::zeros({n_channels, 1}, torch::TensorOptions().dtype(torch::kFloat32));
    at::Tensor ones = torch::ones({n_channels, n_frames + n_order - 1});
    at::Tensor negones = torch::ones({n_channels, n_frames + n_order -1 }) * -1;

    for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {

      // calculate the output at time i_frame
      // initialize it!
      o0 = torch::zeros({n_channels, 1}, torch::TensorOptions().dtype(torch::kFloat32));

      for (int i_offset = 0; i_offset < n_order; ++i_offset) {

        at::Tensor const & input_time_offset = padded_waveform.slice(0, 0, n_channels).slice(1, i_frame + n_order - 1 - i_offset, i_frame + n_order - 1 - i_offset + 1);
        at::Tensor const & output_time_offset = padded_output_waveform.slice(0, 0, n_channels).slice(1, i_frame + n_order - 1 - i_offset, i_frame+ n_order - 1 - i_offset + 1);
        at::Tensor const & a_coeffs_offset = a_coeffs_filled.slice(0, 0, n_channels).slice(1, i_offset, i_offset + 1);
        at::Tensor const & b_coeffs_offset = b_coeffs_filled.slice(0, 0, n_channels).slice(1, i_offset, i_offset + 1);        

        // Multiply by the coefficients
        o0 = torch::addcmul(o0, input_time_offset, b_coeffs_offset, 1);
        if (i_offset != n_order - 1) {
          o0 = torch::addcmul(o0, output_time_offset, a_coeffs_offset, -1);
        }
      }
      // normalize by a0
      o0 = torch::div(o0, a_coeffs_filled.slice(0, 0, n_channels).slice(1, 0, 1));

      // Set the output
      padded_output_waveform.slice(0, 0, n_channels).slice(1, i_frame + n_order - 1, i_frame + n_order - 1 + 1) = o0;
      
    }

    // return clipped, and without initial conditions
    return torch::min(ones, torch::max(negones, padded_output_waveform)).slice(0, 0, n_channels).slice(1, n_order - 1, n_order + n_frames - 1);
  }


  // N.B. TBD - Test with CUDA
  at::Tensor lfilter(
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

#ifdef WITH_CUDA
    auto input_accessor = waveform.packed_accessor64<float,2>();
    auto output_accessor = output_waveform.packed_accessor64<float,2>();    
    auto a_coeffs_accessor = a_coeffs.packed_accessor64<float,1>();
    auto b_coeffs_accessor = b_coeffs.packed_accessor64<float,1>();
#endif
#ifndef WITH_CUDA
    auto input_accessor = waveform.accessor<float,2>();
    auto output_accessor = output_waveform.accessor<float,2>();    
    auto a_coeffs_accessor = a_coeffs.accessor<float,1>();
    auto b_coeffs_accessor = b_coeffs.accessor<float,1>();
#endif

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
      "lfilter_tensor",
      &torch::audio::lfilter_tensor,
      "Executes difference equation with tensor");
  m.def(
      "lfilter_tensor_matrix",
      &torch::audio::lfilter_tensor_matrix,
      "Executes difference equation with tensor");
  m.def(
      "lfilter",
      &torch::audio::lfilter,
      R"mydelimiter(lfilter(waveform, a_coeffs, b_coeffs)

    Performs an IIR filter by evaluating difference equation.

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(n_channel, n_frames)`
        a_coeffs (torch.Tensor): denominator coefficients of difference equation of dimension of `(n_order + 1)`.  Lower delays coefficients are first, e.g. `[a0, a1, a2, ...]`.  Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (torch.Tensor): numerator coefficients of difference equation of dimension of `(n_order + 1)`.  Lower delays coefficients are first, e.g. `[b0, b1, b2, ...]`. Must be same size as a_coeffs (pad with 0's as necessary).
        
    Returns:
        output_waveform (torch.Tensor): Dimension of `(n_channel, n_frames)`

      )mydelimiter");      
#ifdef WITH_CUDA
  m.attr("CUDA_VERSION") = CUDA_VERSION;
#endif

}
