#include <torchaudio/csrc/sox/types.h>
#include <torchaudio/csrc/sox/effects.h>
#include <torchaudio/csrc/sox/effects_chain.h>
#include <torchaudio/csrc/sox/io.h>
#include <torchaudio/csrc/sox/utils.h>

using namespace torch::indexing;
using namespace torchaudio::sox_utils;

namespace torchaudio {
namespace sox_io {

std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> get_info_file(
    const std::string& path,
    c10::optional<std::string>& format) {
  SoxFormat sf(sox_open_read(
      path.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/format.has_value() ? format.value().c_str() : nullptr));

  if (static_cast<sox_format_t*>(sf) == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }

  return std::make_tuple(
      static_cast<int64_t>(sf->signal.rate),
      static_cast<int64_t>(sf->signal.length / sf->signal.channels),
      static_cast<int64_t>(sf->signal.channels),
      static_cast<int64_t>(sf->encoding.bits_per_sample),
      get_encoding(sf->encoding.encoding));
}

namespace {

std::vector<std::vector<std::string>> get_effects(
    c10::optional<int64_t>& frame_offset,
    c10::optional<int64_t>& num_frames) {
  const auto offset = frame_offset.value_or(0);
  if (offset < 0) {
    throw std::runtime_error(
        "Invalid argument: frame_offset must be non-negative.");
  }
  const auto frames = num_frames.value_or(-1);
  if (frames == 0 || frames < -1) {
    throw std::runtime_error(
        "Invalid argument: num_frames must be -1 or greater than 0.");
  }

  std::vector<std::vector<std::string>> effects;
  if (frames != -1) {
    std::ostringstream os_offset, os_frames;
    os_offset << offset << "s";
    os_frames << "+" << frames << "s";
    effects.emplace_back(
        std::vector<std::string>{"trim", os_offset.str(), os_frames.str()});
  } else if (offset != 0) {
    std::ostringstream os_offset;
    os_offset << offset << "s";
    effects.emplace_back(std::vector<std::string>{"trim", os_offset.str()});
  }
  return effects;
}

} // namespace

std::tuple<torch::Tensor, int64_t> load_audio_file(
    const std::string& path,
    c10::optional<int64_t>& frame_offset,
    c10::optional<int64_t>& num_frames,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
    c10::optional<std::string>& format) {
  auto effects = get_effects(frame_offset, num_frames);
  return torchaudio::sox_effects::apply_effects_file(
      path, effects, normalize, channels_first, format);
}

void save_audio_file(
    const std::string& path,
    torch::Tensor tensor,
    int64_t sample_rate,
    bool channels_first,
    c10::optional<double>& compression,
    c10::optional<std::string>& format,
    c10::optional<std::string>& encoding,
    c10::optional<int64_t>& bits_per_sample) {
  validate_input_tensor(tensor);

  const auto filetype = [&]() {
    if (format.has_value())
      return format.value();
    return get_filetype(path);
  }();

  if (filetype == "amr-nb") {
    const auto num_channels = tensor.size(channels_first ? 0 : 1);
    TORCH_CHECK(
        num_channels == 1, "amr-nb format only supports single channel audio.");
  }
  const auto signal_info =
      get_signalinfo(&tensor, sample_rate, filetype, channels_first);
  const auto encoding_info = get_encodinginfo_for_save(
      filetype, tensor.dtype(), compression, encoding, bits_per_sample);

  SoxFormat sf(sox_open_write(
      path.c_str(),
      &signal_info,
      &encoding_info,
      /*filetype=*/filetype.c_str(),
      /*oob=*/nullptr,
      /*overwrite_permitted=*/nullptr));

  if (static_cast<sox_format_t*>(sf) == nullptr) {
    throw std::runtime_error("Error saving audio file: failed to open file.");
  }

  torchaudio::sox_effects_chain::SoxEffectsChain chain(
      /*input_encoding=*/get_tensor_encodinginfo(tensor.dtype()),
      /*output_encoding=*/sf->encoding);
  chain.addInputTensor(&tensor, sample_rate, channels_first);
  chain.addOutputFile(sf);
  chain.run();
}

#ifdef TORCH_API_INCLUDE_EXTENSION_H

std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> get_info_fileobj(
    py::object fileobj,
    c10::optional<std::string>& format) {
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
  auto capacity = 4096;
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

  // In case of streamed data, length can be 0
  validate_input_file(sf, /*check_length=*/false);

  return std::make_tuple(
      static_cast<int64_t>(sf->signal.rate),
      static_cast<int64_t>(sf->signal.length / sf->signal.channels),
      static_cast<int64_t>(sf->signal.channels),
      static_cast<int64_t>(sf->encoding.bits_per_sample),
      get_encoding(sf->encoding.encoding));
}

std::tuple<torch::Tensor, int64_t> load_audio_fileobj(
    py::object fileobj,
    c10::optional<int64_t>& frame_offset,
    c10::optional<int64_t>& num_frames,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
    c10::optional<std::string>& format) {
  auto effects = get_effects(frame_offset, num_frames);
  return torchaudio::sox_effects::apply_effects_fileobj(
      fileobj, effects, normalize, channels_first, format);
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
  AutoReleaseBuffer& operator=(const AutoReleaseBuffer& other) = delete;
  AutoReleaseBuffer& operator=(AutoReleaseBuffer&& other) = delete;
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
    c10::optional<double>& compression,
    c10::optional<std::string>& format,
    c10::optional<std::string>& encoding,
    c10::optional<int64_t>& bits_per_sample) {
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
  }
  const auto signal_info =
      get_signalinfo(&tensor, sample_rate, filetype, channels_first);
  const auto encoding_info = get_encodinginfo_for_save(
      filetype, tensor.dtype(), compression, encoding, bits_per_sample);

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

  torchaudio::sox_effects_chain::SoxEffectsChain chain(
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

#endif // TORCH_API_INCLUDE_EXTENSION_H

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::sox_io_get_info", &torchaudio::sox_io::get_info_file);
  m.def(
      "torchaudio::sox_io_load_audio_file",
      &torchaudio::sox_io::load_audio_file);
  m.def(
      "torchaudio::sox_io_save_audio_file",
      &torchaudio::sox_io::save_audio_file);
}

} // namespace sox_io
} // namespace torchaudio
