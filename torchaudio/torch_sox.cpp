#include <torch/torch.h>

#include <sox.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace torch {
namespace audio {
namespace {
/// Helper struct to safely close the sox_format_t descriptor.
struct SoxDescriptor {
  explicit SoxDescriptor(sox_format_t* fd) noexcept : fd_(fd) {}
  SoxDescriptor(const SoxDescriptor& other) = delete;
  SoxDescriptor(SoxDescriptor&& other) = delete;
  SoxDescriptor& operator=(const SoxDescriptor& other) = delete;
  SoxDescriptor& operator=(SoxDescriptor&& other) = delete;
  ~SoxDescriptor() {
    sox_close(fd_);
  }
  sox_format_t* operator->() noexcept {
    return fd_;
  }
  sox_format_t* get() noexcept {
    return fd_;
  }

 private:
  sox_format_t* fd_;
};

void read_audio(
    SoxDescriptor& fd,
    at::Tensor output,
    int64_t number_of_channels,
    int64_t buffer_length,
    int64_t offset) {
  std::vector<sox_sample_t> buffer(buffer_length);
  if (sox_seek(fd.get(), offset, 0) == SOX_EOF) {
    throw std::runtime_error("sox_seek reached EOF, try reducing offset or num_samples");
  }
  const int64_t samples_read = sox_read(fd.get(), buffer.data(), buffer_length);
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
}

int64_t write_audio(SoxDescriptor& fd, at::Tensor tensor) {
  std::vector<sox_sample_t> buffer(tensor.numel());

  AT_DISPATCH_ALL_TYPES(tensor.type(), "write_audio_buffer", [&] {
    auto* data = tensor.data<scalar_t>();
    std::copy(data, data + tensor.numel(), buffer.begin());
  });

  const auto samples_written =
      sox_write(fd.get(), buffer.data(), buffer.size());

  return samples_written;
}
} // namespace

int read_audio_file(
    const std::string& file_name,
    at::Tensor output,
    int64_t nframes,
    int64_t offset) {
  SoxDescriptor fd(sox_open_read(
      file_name.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));
  if (fd.get() == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }

  const int64_t number_of_channels = fd->signal.channels;
  const int sample_rate = fd->signal.rate;
  const int64_t total_length = fd->signal.length;
  if (total_length == 0) {
    throw std::runtime_error("Error reading audio file: unknown length");
  }

  // calculate buffer length
  int64_t buffer_length = total_length;
  if (offset > 0 && offset < total_length) {
      buffer_length -= offset;
  }
  if (nframes != -1 && buffer_length > nframes) {
      // get requested number of frames
      buffer_length = nframes;
  }

  // buffer length and offset need to be multipled by the number of channels
  buffer_length *= number_of_channels;
  offset *= number_of_channels;

  read_audio(fd, output, number_of_channels, buffer_length, offset);

  return sample_rate;
}

void write_audio_file(
    const std::string& file_name,
    at::Tensor tensor,
    const std::string& extension,
    int sample_rate,
    int precision) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(
        "Error writing audio file: input tensor must be contiguous");
  }

  sox_signalinfo_t signal;
  signal.rate = sample_rate;
  signal.channels = tensor.size(1);
  signal.length = tensor.numel();
  signal.precision = precision; // precision in bits

#if SOX_LIB_VERSION_CODE >= 918272 // >= 14.3.0
  signal.mult = nullptr;
#endif

  SoxDescriptor fd(sox_open_write(
      file_name.c_str(),
      &signal,
      /*encoding=*/nullptr,
      extension.c_str(),
      /*filetype=*/nullptr,
      /*oob=*/nullptr));

  if (fd.get() == nullptr) {
    throw std::runtime_error(
        "Error writing audio file: could not open file for writing");
  }

  const auto samples_written = write_audio(fd, tensor);

  if (samples_written != tensor.numel()) {
    throw std::runtime_error(
        "Error writing audio file: could not write entire buffer");
  }
}

std::tuple<int64_t, int64_t, int64_t, int64_t> get_info(
    const std::string& file_name
  ) {
  SoxDescriptor fd(sox_open_read(
      file_name.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));
  if (fd.get() == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }
  int64_t nchannels = fd->signal.channels;
  int64_t length = fd->signal.length;
  int64_t sample_rate = fd->signal.rate;
  int64_t precision = fd->signal.precision;
  return std::make_tuple(nchannels, length, sample_rate, precision);
}
} // namespace audio
} // namespace torch

PYBIND11_MODULE(_torch_sox, m) {
  m.def(
      "read_audio_file",
      &torch::audio::read_audio_file,
      "Reads an audio file into a tensor");
  m.def(
      "write_audio_file",
      &torch::audio::write_audio_file,
      "Writes data from a tensor into an audio file");
  m.def(
      "get_info",
      &torch::audio::get_info,
      "Gets information about an audio file");
}
