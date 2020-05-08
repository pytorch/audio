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
namespace {

struct SignalInfo : torch::CustomClassHolder {
  const int64_t channels;
  const double rate;
  const int64_t precision;
  const int64_t length;

  explicit SignalInfo(sox_signalinfo_t i) noexcept
      : channels(i.channels),
        rate(i.rate),
        precision(i.precision),
        length(i.length) {}
};

static auto registerSignalInfo =
    torch::class_<SignalInfo>("torchaudio", "SoxSignalInfo")
        .def(
            "GetChannels",
            [](const c10::intrusive_ptr<SignalInfo>& self) -> int64_t {
              return self->channels;
            })
        .def(
            "GetRate",
            [](const c10::intrusive_ptr<SignalInfo>& self) -> double {
              return self->rate;
            })
        .def(
            "GetPrecision",
            [](const c10::intrusive_ptr<SignalInfo>& self) -> int64_t {
              return self->precision;
            })
        .def(
            "GetLength",
            [](const c10::intrusive_ptr<SignalInfo>& self) -> int64_t {
              return self->length;
            });

struct EncodingInfo : torch::CustomClassHolder {
  const int64_t encoding;
  const int64_t bits_per_sample;
  const double compression;
  const int64_t reverse_bytes;
  const int64_t reverse_nibbles;
  const int64_t reverse_bits;
  const bool opposite_endian;

  explicit EncodingInfo(sox_encodinginfo_t i) noexcept
      : encoding((int64_t)i.encoding),
        bits_per_sample(i.bits_per_sample),
        compression(i.compression),
        reverse_bytes((int64_t)i.reverse_bytes),
        reverse_nibbles((int64_t)i.reverse_nibbles),
        reverse_bits((int64_t)i.reverse_bits),
        opposite_endian(i.opposite_endian) {}
};
static auto registerEncodingInfo =
  torch::class_<EncodingInfo>("torchaudio", "SoxEncodingInfo")
  .def("GetEncoding",
       [](const c10::intrusive_ptr<EncodingInfo>& self) -> int64_t {
         return self->encoding;
       })
  .def("GetBPS",
       [](const c10::intrusive_ptr<EncodingInfo>& self) -> int64_t {
         return self->bits_per_sample;
       })
  .def("GetCompression",
       [](const c10::intrusive_ptr<EncodingInfo>& self) -> double {
         return self->compression;
       })
  .def("GetReverseBytes",
       [](const c10::intrusive_ptr<EncodingInfo>& self) -> int64_t {
         return self->reverse_bytes;
       })
  .def("GetReverseNibbles",
       [](const c10::intrusive_ptr<EncodingInfo>& self) -> int64_t {
         return self->reverse_nibbles;
       })
  .def("GetReverseBits",
       [](const c10::intrusive_ptr<EncodingInfo>& self) -> int64_t {
         return self->reverse_bits;
       })
  .def("GetOppositeEndian",
       [](const c10::intrusive_ptr<EncodingInfo>& self) -> bool {
         return self->opposite_endian;
       })
  ;

struct Info : torch::CustomClassHolder {
  SignalInfo signal_info;
  EncodingInfo encoding_info;
  explicit Info(const SignalInfo si, const EncodingInfo ei)
      : signal_info(si), encoding_info(ei) {}
};
static auto registerInfo = torch::class_<Info>("torchaudio", "SoxInfo")
                               .def(
                                   "GetSignalInfo",
                                   [](const c10::intrusive_ptr<Info>& self)
                                       -> c10::intrusive_ptr<SignalInfo> {
                                     return c10::make_intrusive<SignalInfo>(
                                         std::move(self->signal_info));
                                   })
                               .def(
                                   "GetEncodingInfo",
                                   [](const c10::intrusive_ptr<Info>& self)
                                       -> c10::intrusive_ptr<EncodingInfo> {
                                     return c10::make_intrusive<EncodingInfo>(
                                         std::move(self->encoding_info));
                                   });

c10::intrusive_ptr<Info> get_info(const std::string& file_name) {
  SoxDescriptor fd(sox_open_read(
      file_name.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));
  if (fd.get() == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }
  return c10::make_intrusive<Info>(
      std::move(SignalInfo(fd->signal)), std::move(EncodingInfo(fd->encoding)));
}

static auto registerOps = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "torchaudio::sox_get_info(str path) -> __torch__.torch.classes.torchaudio.SoxInfo info")
        .catchAllKernel<decltype(get_info), &get_info>());

} // namespace
} // namespace sox_io
} // namespace torchaudio
