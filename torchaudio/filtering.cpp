#include <torch/extension.h>

#include <sox.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace torch {
namespace audio {

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

}
