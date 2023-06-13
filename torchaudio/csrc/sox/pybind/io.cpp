#include <torchaudio/csrc/sox/io.h>
#include <torchaudio/csrc/sox/pybind/effects.h>
#include <torchaudio/csrc/sox/pybind/effects_chain.h>
#include <torchaudio/csrc/sox/pybind/io.h>
#include <torchaudio/csrc/sox/pybind/utils.h>
#include <torchaudio/csrc/sox/types.h>

#include <utility>

namespace torchaudio::sox {

auto get_info_fileobj(py::object fileobj, c10::optional<std::string> format)
    -> c10::optional<MetaDataTuple> {
  // Prepare in-memory file object
  // When libsox opens a file, it also reads the header.
  // When opening a file there are two functions that might touch FILE* (and the
  // underlying buffer).
  // * `auto_detect_format`
  //   https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/formats.c#L43
  // * `startread` handler of detected format.
  //   https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/formats.c#L574
  // To see the handler of a particular format, go to
  //   https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/<FORMAT>.c
  // For example, voribs can be found
  //   https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/vorbis.c#L97-L158
  //
  // `auto_detect_format` function only requires 256 bytes, but format-dependent
  // `startread` handler might require more data. In case of vorbis, the size of
  // header is unbounded, but typically 4kB maximum.
  //
  // "The header size is unbounded, although for streaming a rule-of-thumb of
  // 4kB or less is recommended (and Xiph.Org's Vorbis encoder follows this
  // suggestion)."
  //
  // See:
  // https://xiph.org/vorbis/doc/Vorbis_I_spec.html
  const auto capacity = [&]() {
    // NOTE:
    // Use the abstraction provided by `libtorchaudio` to access the global
    // config defined by libsox. Directly using `sox_get_globals` function will
    // end up retrieving the static variable defined in `_torchaudio`, which is
    // not correct.
    const auto bufsiz = get_buffer_size();
    const int64_t kDefaultCapacityInBytes = 4096;
    return (bufsiz > kDefaultCapacityInBytes) ? bufsiz
                                              : kDefaultCapacityInBytes;
  }();
  std::string buffer(capacity, '\0');
  auto* buf = const_cast<char*>(buffer.data());
  auto num_read = read_fileobj(&fileobj, capacity, buf);
  // If the file is shorter than 256, then libsox cannot read the header.
  auto buf_size = (num_read > 256) ? num_read : 256;

  SoxFormat sf(sox_open_mem_read(
      buf,
      buf_size,
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/format.has_value() ? format.value().c_str() : nullptr));

  if (static_cast<sox_format_t*>(sf) == nullptr ||
      sf->encoding.encoding == SOX_ENCODING_UNKNOWN) {
    return c10::optional<MetaDataTuple>{};
  }

  return std::forward_as_tuple(
      static_cast<int64_t>(sf->signal.rate),
      static_cast<int64_t>(sf->signal.length / sf->signal.channels),
      static_cast<int64_t>(sf->signal.channels),
      static_cast<int64_t>(sf->encoding.bits_per_sample),
      get_encoding(sf->encoding.encoding));
}

auto load_audio_fileobj(
    py::object fileobj,
    c10::optional<int64_t> frame_offset,
    c10::optional<int64_t> num_frames,
    c10::optional<bool> normalize,
    c10::optional<bool> channels_first,
    c10::optional<std::string> format)
    -> c10::optional<std::tuple<torch::Tensor, int64_t>> {
  auto effects = get_effects(frame_offset, num_frames);
  return apply_effects_fileobj(
      std::move(fileobj),
      effects,
      normalize,
      channels_first,
      std::move(format));
}

namespace {

// helper class to automatically release buffer, to be used by
// save_audio_fileobj
struct AutoReleaseBuffer {
  char* ptr;
  size_t size;

  AutoReleaseBuffer() : ptr(nullptr), size(0) {}
  AutoReleaseBuffer(const AutoReleaseBuffer& other) = delete;
  AutoReleaseBuffer(AutoReleaseBuffer&& other) = delete;
  auto operator=(const AutoReleaseBuffer& other) -> AutoReleaseBuffer& = delete;
  auto operator=(AutoReleaseBuffer&& other) -> AutoReleaseBuffer& = delete;
  ~AutoReleaseBuffer() {
    if (ptr) {
      free(ptr);
    }
  }
};

} // namespace

void save_audio_fileobj(
    py::object fileobj,
    torch::Tensor tensor,
    int64_t sample_rate,
    bool channels_first,
    c10::optional<double> compression,
    c10::optional<std::string> format,
    c10::optional<std::string> encoding,
    c10::optional<int64_t> bits_per_sample) {
  validate_input_tensor(tensor);

  if (!format.has_value()) {
    throw std::runtime_error(
        "`format` is required when saving to file object.");
  }
  const auto filetype = format.value();

  if (filetype == "amr-nb") {
    const auto num_channels = tensor.size(channels_first ? 0 : 1);
    if (num_channels != 1) {
      throw std::runtime_error(
          "amr-nb format only supports single channel audio.");
    }
  } else if (filetype == "htk") {
    const auto num_channels = tensor.size(channels_first ? 0 : 1);
    if (num_channels != 1) {
      throw std::runtime_error(
          "htk format only supports single channel audio.");
    }
  } else if (filetype == "gsm") {
    const auto num_channels = tensor.size(channels_first ? 0 : 1);
    if (num_channels != 1) {
      throw std::runtime_error(
          "gsm format only supports single channel audio.");
    }
    if (sample_rate != 8000) {
      throw std::runtime_error(
          "gsm format only supports a sampling rate of 8kHz.");
    }
  }
  const auto signal_info =
      get_signalinfo(&tensor, sample_rate, filetype, channels_first);
  const auto encoding_info = get_encodinginfo_for_save(
      filetype,
      tensor.dtype(),
      compression,
      std::move(encoding),
      bits_per_sample);

  AutoReleaseBuffer buffer;

  SoxFormat sf(sox_open_memstream_write(
      &buffer.ptr,
      &buffer.size,
      &signal_info,
      &encoding_info,
      filetype.c_str(),
      /*oob=*/nullptr));

  if (static_cast<sox_format_t*>(sf) == nullptr) {
    throw std::runtime_error(
        "Error saving audio file: failed to open memory stream.");
  }

  SoxEffectsChainPyBind chain(
      /*input_encoding=*/get_tensor_encodinginfo(tensor.dtype()),
      /*output_encoding=*/sf->encoding);
  chain.addInputTensor(&tensor, sample_rate, channels_first);
  chain.addOutputFileObj(sf, &buffer.ptr, &buffer.size, &fileobj);
  chain.run();

  // Closing the sox_format_t is necessary for flushing the last chunk to the
  // buffer
  sf.close();

  fileobj.attr("write")(py::bytes(buffer.ptr, buffer.size));
}

} // namespace torchaudio::sox
