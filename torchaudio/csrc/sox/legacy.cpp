#include <torchaudio/csrc/sox/legacy.h>

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
    if (fd_ != nullptr) {
      sox_close(fd_);
    }
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

int64_t write_audio(SoxDescriptor& fd, at::Tensor tensor) {
  std::vector<sox_sample_t> buffer(tensor.numel());

  AT_DISPATCH_ALL_TYPES(tensor.scalar_type(), "write_audio_buffer", [&] {
    auto* data = tensor.data_ptr<scalar_t>();
    std::copy(data, data + tensor.numel(), buffer.begin());
  });

  const auto samples_written =
      sox_write(fd.get(), buffer.data(), buffer.size());

  return samples_written;
}

void read_audio(SoxDescriptor& fd, at::Tensor output, int64_t buffer_length) {
  std::vector<sox_sample_t> buffer(buffer_length);

  int number_of_channels = fd->signal.channels;
  const int64_t samples_read = sox_read(fd.get(), buffer.data(), buffer_length);
  if (samples_read == 0) {
    throw std::runtime_error(
        "Error reading audio file: empty file or read failed in sox_read");
  }

  output.resize_({samples_read / number_of_channels, number_of_channels});
  output = output.contiguous();

  AT_DISPATCH_ALL_TYPES(output.scalar_type(), "read_audio_buffer", [&] {
    auto* data = output.data_ptr<scalar_t>();
    std::copy(buffer.begin(), buffer.begin() + samples_read, data);
  });
}
} // namespace

std::tuple<sox_signalinfo_t, sox_encodinginfo_t> get_info(
    const std::string& file_name) {
  SoxDescriptor fd(sox_open_read(
      file_name.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));
  if (fd.get() == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }
  return std::make_tuple(fd->signal, fd->encoding);
}

int read_audio_file(
    const std::string& file_name,
    at::Tensor output,
    bool ch_first,
    int64_t nframes,
    int64_t offset,
    sox_signalinfo_t* si,
    sox_encodinginfo_t* ei,
    const char* ft) {
  SoxDescriptor fd(sox_open_read(file_name.c_str(), si, ei, ft));
  if (fd.get() == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }

  // signal info

  const int number_of_channels = fd->signal.channels;
  const int sample_rate = fd->signal.rate;
  const int64_t total_length = fd->signal.length;

  // multiply offset and number of frames by number of channels
  offset *= number_of_channels;
  nframes *= number_of_channels;

  if (total_length == 0) {
    throw std::runtime_error("Error reading audio file: unknown length");
  }
  if (offset > total_length) {
    throw std::runtime_error("Offset past EOF");
  }

  // calculate buffer length
  int64_t buffer_length = total_length;
  if (offset > 0) {
    buffer_length -= offset;
  }
  if (nframes > 0 && buffer_length > nframes) {
    buffer_length = nframes;
  }

  // seek to offset point before reading data
  if (sox_seek(fd.get(), offset, 0) == SOX_EOF) {
    throw std::runtime_error(
        "sox_seek reached EOF, try reducing offset or num_samples");
  }

  // read data and fill output tensor
  read_audio(fd, output, buffer_length);

  // L x C -> C x L, if desired
  if (ch_first) {
    output.transpose_(1, 0);
  }

  return sample_rate;
}

void write_audio_file(
    const std::string& file_name,
    const at::Tensor& tensor,
    sox_signalinfo_t* si,
    sox_encodinginfo_t* ei,
    const char* file_type) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(
        "Error writing audio file: input tensor must be contiguous");
  }

#if SOX_LIB_VERSION_CODE >= 918272 // >= 14.3.0
  si->mult = nullptr;
#endif

  SoxDescriptor fd(sox_open_write(
      file_name.c_str(),
      si,
      ei,
      file_type,
      /*oob=*/nullptr,
      /*overwrite=*/nullptr));

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

} // namespace audio
} // namespace torch
