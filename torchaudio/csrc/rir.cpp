/*
Copyright (c) 2014-2017 EPFL-LCAV
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/**
 * Image source method implementation based on PyRoomAcoustics:
 * https://github.com/LCAV/pyroomacoustics
 */
#include <torch/script.h>
#include <torch/torch.h>
#include <cmath>
using namespace torch::indexing;

namespace torchaudio {
namespace rir {

namespace {
/**
 * @brief Sum up impulse response signal of all image sources into one Tensor
 * based on delays of arrival of the image sources. The implementation is based
 * on the one in pyroomacoustics:
 * https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/build_rir.pyx
 *
 * @tparam scalar_t The type of irs and rirs Tensor
 * @param irs The impulse responses for all image sources. Tensor with
 * dimensions `(num_band, num_image, num_mic, ir_length)`.
 * @param delay The delays for the impulse response of each image source. Tensor
 * with dimensions `(num_inage, num_mic)`.
 * @param rirs The output room impulse response signal. Tensor with dimensions
 * `(num_band, num_mic, rir_length)`.
 * @param num_band The number of frequency bands for the wall materials.
 * @param num_image The number of image sources in irs.
 * @param num_mic The number of microphones in the array.
 * @param ir_length The length of impulse response signal.
 */
template <typename scalar_t>
void simulate_rir_impl(
    const torch::Tensor& irs,
    const torch::Tensor& delay,
    const int64_t rir_length,
    const int64_t num_band,
    const int64_t num_image,
    const int64_t num_mic,
    const int64_t ir_length,
    torch::Tensor& rirs) {
  const scalar_t* input_data = irs.data_ptr<scalar_t>();
  const int* delay_data = delay.data_ptr<int>();
  scalar_t* output_data = rirs.data_ptr<scalar_t>();
  for (auto i = 0; i < num_band * num_image * num_mic; i++) {
    int64_t offset_input = i * ir_length;
    int64_t mic = i % num_mic;
    int64_t image = ((i - mic) / num_mic) % num_image;
    int64_t band = (i - mic - image * num_mic) / (num_image * num_mic);
    int64_t offset_output = (band * num_mic + mic) * rir_length;
    int64_t offset_delay = image * num_mic + mic;
    for (auto j = 0; j < ir_length; j++) {
      output_data[offset_output + j + delay_data[offset_delay]] +=
          input_data[offset_input + j];
    }
  }
}

/**
 * @brief Sum up impulse response signal of all image sources into one Tensor
 * based on delays of arrival of the image sources.
 *
 * @param irs The impulse responses for all image sources. Tensor with
 * dimensions `(num_band, num_image, num_mic, ir_length)`.
 * @param delay The delays for the impulse response of each image source. Tensor
 * with dimensions `(num_inage, num_mic)`.
 * @param rir_length The length of the output room impulse response signal.
 * @return torch::Tensor The output room impulse response signal. Tensor with
 * dimensions `(num_band, num_mic, rir_length)`.
 */
torch::Tensor simulate_rir(
    const torch::Tensor& irs,
    const torch::Tensor& delay,
    const int64_t rir_length) {
  const int64_t num_band = irs.size(0);
  const int64_t num_image = irs.size(1);
  const int64_t num_mic = irs.size(2);
  const int64_t ir_length = irs.size(3);
  torch::Tensor rirs =
      torch::zeros({num_band, num_mic, rir_length}, irs.dtype());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(irs.scalar_type(), "build_rir", [&] {
    simulate_rir_impl<scalar_t>(
        irs, delay, rir_length, num_band, num_image, num_mic, ir_length, rirs);
  });
  return rirs;
}

/**
 * @brief Create the band-pass filters for the octave bands.
 * The implementation is based on the one in pyroomacoustics:
 * https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/acoustics.py#L261
 *
 * @tparam scalar_t The type of center frequencies and output filter Tensors.
 * @param centers The Tensor that stores the center frequencies of octave bands.
 * Tensor with dimension `(num_band,)`.
 * @param sample_rate The sample_rate of simulated room impulse response signal.
 * @param n_fft The number of fft points.
 * @param filters The output band-pass filter. Tensor with dimensions
 * `(num_band, n_fft - 1)`.
 */
template <typename scalar_t>
void make_rir_filter_impl(
    torch::Tensor& centers,
    double sample_rate,
    int64_t n_fft,
    torch::Tensor& filters) {
  int64_t n = centers.size(0);
  torch::Tensor new_bands = torch::zeros({n, 2}, centers.dtype());
  scalar_t* newband_data = new_bands.data_ptr<scalar_t>();
  const scalar_t* centers_data = centers.data_ptr<scalar_t>();
  for (int64_t i = 0; i < n; i++) {
    if (i == 0) {
      newband_data[i * 2] = centers_data[0] / 2;
      newband_data[i * 2 + 1] = centers_data[1];
    } else if (i == n - 1) {
      newband_data[i * 2] = centers_data[n - 2];
      newband_data[i * 2 + 1] = sample_rate / 2;
    } else {
      newband_data[i * 2] = centers_data[i - 1];
      newband_data[i * 2 + 1] = centers_data[i + 1];
    }
  }
  const auto half = 0.5;
  auto n_freq = n_fft / 2 + 1;
  torch::Tensor freq_resp = torch::zeros({n_freq, n}, centers.dtype());
  torch::Tensor freq =
      torch::arange(n_freq, centers.dtype()) / n_fft * sample_rate;
  const scalar_t* freq_data = freq.data_ptr<scalar_t>();
  scalar_t* freqreq_data = freq_resp.data_ptr<scalar_t>();

  for (auto i = 0; i < n; i++) {
    for (auto j = 0; j < n_freq; j++) {
      if (freq_data[j] >= newband_data[i * 2] &&
          freq_data[j] < centers_data[i]) {
        freqreq_data[j * n + i] =
            half * (1 + cos(2 * M_PI * freq_data[j] / centers_data[i]));
      }
      if (i != n - 1 && freq_data[j] >= centers_data[i] &&
          freq_data[j] < newband_data[i * 2 + 1]) {
        freqreq_data[j * n + i] =
            half * (1 - cos(2 * M_PI * freq_data[j] / newband_data[i * 2 + 1]));
      }
      if (i == n - 1 && centers_data[i] <= freq_data[j]) {
        freqreq_data[j * n + i] = 1.0;
      }
    }
  }
  filters = torch::fft::fftshift(torch::fft::irfft(freq_resp, n_fft, 0), 0);
  filters = filters.index({Slice(1)}).transpose(0, 1);
}

/**
 * @brief Create the band-pass filters for the octave bands.
 *
 * @param centers The Tensor that stores the center frequencies of octave bands.
 * Tensor with dimension `(num_band,)`.
 * @param sample_rate The sample_rate of simulated room impulse response signal.
 * @param n_fft The number of fft points.
 * @return torch::Tensor The output band-pass filter. Tensor with dimensions
 * `(num_band, n_fft - 1)`.
 */
torch::Tensor make_rir_filter(
    torch::Tensor centers,
    double sample_rate,
    int64_t n_fft) {
  torch::Tensor filters;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      centers.scalar_type(), "make_filter", [&] {
        make_rir_filter_impl<scalar_t>(centers, sample_rate, n_fft, filters);
      });
  return filters;
}

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("torchaudio::_simulate_rir", torchaudio::rir::simulate_rir);
  m.impl("torchaudio::_make_rir_filter", torchaudio::rir::make_rir_filter);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "torchaudio::_simulate_rir(Tensor irs, Tensor delay_i, int rir_length) -> Tensor");
  m.def(
      "torchaudio::_make_rir_filter(Tensor centers, float sample_rate, int n_fft) -> Tensor");
}

} // Anonymous namespace
} // namespace rir
} // namespace torchaudio
