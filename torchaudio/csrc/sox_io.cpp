#include "sox.h"
#include "torch/script.h"

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

namespace torchaudio {
namespace sox_io {

struct SoxAudioInfo : torch::CustomClassHolder {
  // from sox_signalinfo_t
  const int64_t channels;
  const double rate;
  const int64_t precision;
  const int64_t length;
  // from sox_encodinginfo_t
  const int64_t encoding;
  const int64_t bits_per_sample;
  const double compression;
  const int64_t reverse_bytes;
  const int64_t reverse_nibbles;
  const int64_t reverse_bits;
  const bool opposite_endian;

  explicit SoxAudioInfo(sox_signalinfo_t si, sox_encodinginfo_t ei) noexcept
      : channels(si.channels),
        rate(si.rate),
        precision(si.precision),
        length(si.length),
        encoding((int64_t)ei.encoding),
        bits_per_sample(ei.bits_per_sample),
        compression(ei.compression),
        reverse_bytes((int64_t)ei.reverse_bytes),
        reverse_nibbles((int64_t)ei.reverse_nibbles),
        reverse_bits((int64_t)ei.reverse_bits),
        opposite_endian(ei.opposite_endian) {}
};

static auto registerSoxAudioInfo =
    torch::class_<SoxAudioInfo>("torchaudio", "SoxAudioInfo")
        .def(
            "GetChannels",
            [](const c10::intrusive_ptr<SoxAudioInfo>& self) -> int64_t {
              return self->channels;
            })
        .def(
            "GetRate",
            [](const c10::intrusive_ptr<SoxAudioInfo>& self) -> double {
              return self->rate;
            })
        .def(
            "GetPrecision",
            [](const c10::intrusive_ptr<SoxAudioInfo>& self) -> int64_t {
              return self->precision;
            })
        .def(
            "GetLength",
            [](const c10::intrusive_ptr<SoxAudioInfo>& self) -> int64_t {
              return self->length;
            })
        .def(
            "GetEncoding",
            [](const c10::intrusive_ptr<SoxAudioInfo>& self) -> int64_t {
              return self->encoding;
            })
        .def(
            "GetBPS",
            [](const c10::intrusive_ptr<SoxAudioInfo>& self) -> int64_t {
              return self->bits_per_sample;
            })
        .def(
            "GetCompression",
            [](const c10::intrusive_ptr<SoxAudioInfo>& self) -> double {
              return self->compression;
            })
        .def(
            "GetReverseBytes",
            [](const c10::intrusive_ptr<SoxAudioInfo>& self) -> int64_t {
              return self->reverse_bytes;
            })
        .def(
            "GetReverseNibbles",
            [](const c10::intrusive_ptr<SoxAudioInfo>& self) -> int64_t {
              return self->reverse_nibbles;
            })
        .def(
            "GetReverseBits",
            [](const c10::intrusive_ptr<SoxAudioInfo>& self) -> int64_t {
              return self->reverse_bits;
            })
        .def(
            "GetOppositeEndian",
            [](const c10::intrusive_ptr<SoxAudioInfo>& self) -> bool {
              return self->opposite_endian;
            });

/// Reads an audio file from the given `path` and returns a tuple of
/// sox_signalinfo_t and sox_encodinginfo_t, which contain information about
/// the audio file such as sample rate, length, bit precision, encoding and
/// more. Throws `std::runtime_error` if the audio file could not be opened, or
/// an error occurred during reading of the audio data.
c10::intrusive_ptr<SoxAudioInfo> get_info(const std::string& file_name) {
  SoxDescriptor fd(sox_open_read(
      file_name.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));
  if (fd.get() == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }
  return c10::make_intrusive<SoxAudioInfo>(fd->signal, fd->encoding);
}

static auto registerGetInfo = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "torchaudio::sox_get_info(str path) -> __torch__.torch.classes.torchaudio.SoxAudioInfo info")
        .catchAllKernel<decltype(get_info), &get_info>());

} // namespace sox_io
} // namespace torchaudio
