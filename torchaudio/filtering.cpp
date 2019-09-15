#include <torch/extension.h>

//#include <algorithm>
//#include <cstdint>
//#include <stdexcept>
//#include <vector>

namespace torch {
namespace audio {


  void diff_eq(
    at::Tensor& input_waveform,
    at::Tensor& output_waveform,
    at::Tensor& a_coeffs,
    at::Tensor& b_coeffs
  ) {

    int64_t n_channels = input_waveform.size(0);
    int64_t n_frames = input_waveform.size(1);

    assert(output_waveform.size(0) == n_channels);
    assert(output_waveform.size(1) == n_frames);

    int n_order = a_coeffs.size(0); // n'th order filter
    assert(a_coeffs.size(0) == b_coeffs.size(0));

    for (int64_t i_channel = 0; i_channel < n_channels; ++i_channel) {

      // allocate a temporary data structure of size 2 x n_order
      double i_s[n_order]= { };
      double o_s[n_order]= { };
    
      for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {

        // populate the output
        i_s[n_order-1] = input_waveform[i_channel][i_frame].item<double>();
        double o0 = 0;
        for (int i = 0; i < n_order; ++i) {
          o0 += i_s[i] * b_coeffs[n_order - i - 1].item<double>();
          if (i != n_order - 1) {
            o0 -= o_s[i] * a_coeffs[n_order - i - 1].item<double>();
          }
        }
        o0 = o0 / a_coeffs[0].item<double>();
        if (o0 > 1) o0 = 1;
        if (o0 < -1) o0 = -1;
        o_s[n_order-1] = o0;
        output_waveform[i_channel][i_frame] = o0;
        
        // shift everything over
        for (int i = 0; i < (n_order - 1); ++i) {
          i_s[i] = i_s[i+1];
          o_s[i] = o_s[i+1];
        }
      }
    }
  }



  void biquad(
    at::Tensor& input_waveform,
    at::Tensor& output_waveform,
    double b0,
    double b1,
    double b2,
    double a0,
    double a1,
    double a2
  ) {

    int64_t n_channels = input_waveform.size(0);
    int64_t n_frames = input_waveform.size(1);

    //assert(output_waveform.size(0) == n_channels);
    //assert(output_waveform.size(1) == n_frames);

    b0 = b0 / a0;
    b1 = b1 / a0;
    b2 = b2 / a0;
    a1 = a1 / a0;
    a2 = a2 / a0;

    for (int64_t i_channel = 0; i_channel < n_channels; ++i_channel) {

      double pi1 = 0;
      double pi2 = 0;
      double po1 = 0;
      double po2 = 0;
      double o0, i0;
    
      for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {
        i0 = input_waveform[i_channel][i_frame].item<double>();
        o0 = (i0 * b0 + pi1 * b1 + pi2 * b2 - po1 * a1 - po2 * a2);
        
        // increment time forward
        pi2 = pi1;
        pi1 = i0;
        po2 = po1;
        po1 = o0;
        if (o0 > 1) {
          o0 = 1;
        } else if (o0 < -1) {
          o0 = -1;
        }
        output_waveform[i_channel][i_frame] = o0;
      }
    }
  }


}}


PYBIND11_MODULE(_torch_filtering, m) {
  m.def(
      "biquad",
      &torch::audio::biquad,
      "Executes biquad");
  m.def(
      "diff_eq",
      &torch::audio::diff_eq,
      "Executes difference equation");
}
