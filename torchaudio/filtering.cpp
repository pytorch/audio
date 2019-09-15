#include <torch/extension.h>

//#include <algorithm>
//#include <cstdint>
//#include <stdexcept>
//#include <vector>

namespace torch {
namespace audio {

  // N.B. only handles floating point right now
  void diff_eq(
    at::Tensor const & input_waveform,
    at::Tensor& output_waveform,
    at::Tensor const & a_coeffs,
    at::Tensor const & b_coeffs
  ) {

    // assumes waveform is normalized between 1 and -1

    int64_t n_channels = input_waveform.size(0);
    int64_t n_frames = input_waveform.size(1);

    assert(output_waveform.size(0) == n_channels);
    assert(output_waveform.size(1) == n_frames);

    auto input_accessor = input_waveform.accessor<float,2>();
    auto output_accessor = output_waveform.accessor<float,2>();    
    auto a_coeffs_accessor = a_coeffs.accessor<float,1>();
    auto b_coeffs_accessor = b_coeffs.accessor<float,1>();

    int n_order = a_coeffs.size(0); // n'th order - 1 filter
    assert(a_coeffs.size(0) == b_coeffs.size(0));

    for (int64_t i_channel = 0; i_channel < n_channels; ++i_channel) {

      // allocate a temporary data structure of size 2 x (n_order + 1)
      // set to 0 because initial conditions are 0
      float i_s[n_order]= { };
      float o_s[n_order]= { };
    
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
  }

}}


PYBIND11_MODULE(_torch_filtering, m) {
  m.def(
      "diff_eq",
      &torch::audio::diff_eq,
      "Executes difference equation");
}
