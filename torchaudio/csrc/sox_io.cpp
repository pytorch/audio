#include <sox.h>
#include <torchaudio/csrc/sox_io.h>

using namespace torch::indexing;

namespace torchaudio {
namespace sox_io {

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

} // namespace

c10::intrusive_ptr<::torchaudio::SignalInfo> get_info(
    const std::string& file_name) {
  SoxDescriptor sd(sox_open_read(
      file_name.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));

  if (sd.get() == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }

  return c10::make_intrusive<::torchaudio::SignalInfo>(
      static_cast<int64_t>(sd->signal.rate),
      static_cast<int64_t>(sd->signal.channels),
      static_cast<int64_t>(sd->signal.length / sd->signal.channels));
}

} // namespace sox_io
} // namespace torchaudio
