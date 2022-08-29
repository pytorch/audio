#include <math.h>
#include <torch/script.h>
#include <torch/torch.h>
using namespace torch::indexing;

namespace torchaudio {
namespace rir {

template <typename scalar_t>
void build_rir_impl(
    const torch::Tensor& irs,
    const torch::Tensor& delay,
    torch::Tensor& rirs,
    const int64_t rir_length,
    const int64_t num_band,
    const int64_t num_image,
    const int64_t num_mic,
    const int64_t ir_length) {
  const scalar_t* input_data = irs.data_ptr<scalar_t>();
  const int* delay_data = delay.data_ptr<int>();
  scalar_t* output_data = rirs.data_ptr<scalar_t>();
  at::parallel_for(
      0, num_band * num_image * num_mic, 0, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
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
      });
}

torch::Tensor build_rir(
    const torch::Tensor irs,
    const torch::Tensor delay,
    const int64_t rir_length) {
  const int64_t num_band = irs.size(0);
  const int64_t num_image = irs.size(1);
  const int64_t num_mic = irs.size(2);
  const int64_t ir_length = irs.size(3);
  torch::Tensor rirs =
      torch::zeros({num_band, num_mic, rir_length}, irs.dtype());
  rirs.requires_grad_(true);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(irs.scalar_type(), "build_rir", [&] {
    build_rir_impl<scalar_t>(
        irs, delay, rirs, rir_length, num_band, num_image, num_mic, ir_length);
  });
  return rirs;
}

template <typename scalar_t>
void make_filter_impl(
    torch::Tensor& centers,
    double sample_rate,
    int64_t n_fft,
    torch::Tensor& filters) {
  int64_t n = centers.size(0);
  torch::Tensor new_bands = torch::zeros({n, 2}, centers.dtype());
  new_bands.requires_grad_(true);
  scalar_t* newband_data = new_bands.data_ptr<scalar_t>();
  const scalar_t* centers_data = centers.data_ptr<scalar_t>();
  at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
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
  });
  auto n_freq = n_fft / 2 + 1;
  torch::Tensor freq_resp = torch::zeros({n_freq, n}, centers.dtype());
  torch::Tensor freq =
      torch::arange(n_freq, centers.dtype()) / n_fft * sample_rate;
  const scalar_t* freq_data = freq.data_ptr<scalar_t>();
  scalar_t* freqreq_data = freq_resp.data_ptr<scalar_t>();

  at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
    at::parallel_for(0, n_freq, 0, [&](int64_t start2, int64_t end2) {
      for (auto i = start; i < end; i++) {
        for (auto j = start2; j < end2; j++) {
          if (freq_data[j] >= newband_data[i * 2] &&
              freq_data[j] < centers_data[i]) {
            freqreq_data[j * n + i] =
                0.5 * (1 + cos(2 * M_PI * freq_data[j] / centers_data[i]));
          }
          if (i != n - 1 && freq_data[j] >= centers_data[i] &&
              freq_data[j] < newband_data[i * 2 + 1]) {
            freqreq_data[j * n + i] = 0.5 *
                (1 - cos(2 * M_PI * freq_data[j] / newband_data[i * 2 + 1]));
          }
          if (i == n - 1 && centers_data[i] <= freq_data[j]) {
            freqreq_data[j * n + i] = 1.0;
          }
        }
      }
    });
  });
  filters = torch::fft::fftshift(torch::fft::irfft(freq_resp, n_fft, 0), 0);
  filters = filters.index({Slice(1)}).transpose(0, 1);
}

torch::Tensor make_filter(
    torch::Tensor centers,
    double sample_rate,
    int64_t n_fft) {
  torch::Tensor filters;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      centers.scalar_type(), "make_filter", [&] {
        make_filter_impl<scalar_t>(centers, sample_rate, n_fft, filters);
      });
  return filters;
}

TORCH_LIBRARY(rir, m) {
  m.def(
      "rir::build_rir(Tensor irs, Tensor delay_i, int rir_length) -> Tensor",
      &torchaudio::rir::build_rir);
  m.def("rir::make_filter", &torchaudio::rir::make_filter);
}

} // namespace rir
} // namespace torchaudio
