#include <torch/torch.h>

#include <sox.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace torch { namespace audio {
int read_audio_file(const std::string& file_name, at::Tensor output) {
  if (sox_init() != SOX_SUCCESS) {
    throw std::runtime_error("Error initializing sox library");
  }
  sox_format_t* fd = sox_open_read(
      file_name.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr);
  if (fd == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }

  const int64_t number_of_channels = fd->signal.channels;
  const int sample_rate = fd->signal.rate;
  const int64_t buffer_length = fd->signal.length;
  if (buffer_length == 0) {
    throw std::runtime_error("Error reading audio file: unknown length");
  }

  std::vector<sox_sample_t> buffer(buffer_length);
  const int64_t samples_read = sox_read(fd, buffer.data(), buffer_length);
  if (samples_read == 0) {
    throw std::runtime_error(
        "Error reading audio file: empty file or read failed in sox_read");
  }

  output.resize_({samples_read / number_of_channels, number_of_channels});
  output = output.contiguous();

  AT_DISPATCH_ALL_TYPES(output.type(), "read_audio_buffer", [&] {
    auto* data = output.data<scalar_t>();
    std::copy(buffer.begin(), buffer.begin() + samples_read, data);
  });

  sox_quit();

  return sample_rate;
}

void write_audio_file(
    const std::string& file_name,
    at::Tensor tensor,
    const std::string& extension,
    int sample_rate) {
  if (sox_init() != SOX_SUCCESS) {
    throw std::runtime_error("Error initializing sox library");
  }

  if (!tensor.is_contiguous()) {
    throw std::runtime_error(
        "Error writing audio file: input tensor must be contiguous");
  }

  sox_signalinfo_t signal;
  signal.rate = sample_rate;
  signal.channels = tensor.size(1);
  signal.length = tensor.numel();
  signal.precision = 32; // precision in bits

#if SOX_LIB_VERSION_CODE >= 918272 // >= 14.3.0
  signal.mult = nullptr;
#endif

  sox_format_t* fd = sox_open_write(
      file_name.c_str(),
      &signal,
      /*encoding=*/nullptr,
      extension.c_str(),
      /*filetype=*/nullptr,
      /*oob=*/nullptr);

  if (fd == nullptr) {
    throw std::runtime_error(
        "Error writing audio file: could not open file for writing");
  }

  std::vector<sox_sample_t> buffer(tensor.numel());

  AT_DISPATCH_ALL_TYPES(tensor.type(), "write_audio_buffer", [&] {
    auto* data = tensor.data<scalar_t>();
    std::copy(data, data + tensor.numel(), buffer.begin());
  });

  const auto samples_written = sox_write(fd, buffer.data(), buffer.size());

  // Free buffer and sox structures (before possibly throwing).
  sox_close(fd);
  sox_quit();

  if (samples_written != buffer.size()) {
    throw std::runtime_error(
        "Error writing audio file: could not write entire buffer");
  }
}
}} // namespace torch::audio

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "read_audio_file",
      &torch::audio::read_audio_file,
      "Reads an audio file into a tensor");
  m.def(
      "write_audio_file",
      &torch::audio::write_audio_file,
      "Writes data from a tensor into an audio file");
}
