#include <sox.h>
#include <torchaudio/csrc/sox/effects.h>
#include <torchaudio/csrc/sox/effects_chain.h>
#include <torchaudio/csrc/sox/utils.h>

using namespace torchaudio::sox_utils;

namespace torchaudio {
namespace sox_effects {

namespace {

enum SoxEffectsResourceState { NotInitialized, Initialized, ShutDown };
SoxEffectsResourceState SOX_RESOURCE_STATE = NotInitialized;
std::mutex SOX_RESOUCE_STATE_MUTEX;

} // namespace

void initialize_sox_effects() {
  const std::lock_guard<std::mutex> lock(SOX_RESOUCE_STATE_MUTEX);

  switch (SOX_RESOURCE_STATE) {
    case NotInitialized:
      if (sox_init() != SOX_SUCCESS) {
        throw std::runtime_error("Failed to initialize sox effects.");
      };
      SOX_RESOURCE_STATE = Initialized;
    case Initialized:
      break;
    case ShutDown:
      throw std::runtime_error(
          "SoX Effects has been shut down. Cannot initialize again.");
  }
};

void shutdown_sox_effects() {
  const std::lock_guard<std::mutex> lock(SOX_RESOUCE_STATE_MUTEX);

  switch (SOX_RESOURCE_STATE) {
    case NotInitialized:
      throw std::runtime_error(
          "SoX Effects is not initialized. Cannot shutdown.");
    case Initialized:
      if (sox_quit() != SOX_SUCCESS) {
        throw std::runtime_error("Failed to initialize sox effects.");
      };
      SOX_RESOURCE_STATE = ShutDown;
    case ShutDown:
      break;
  }
}

std::tuple<torch::Tensor, int64_t> apply_effects_tensor(
    torch::Tensor waveform,
    int64_t sample_rate,
    std::vector<std::vector<std::string>> effects,
    bool channels_first) {
  validate_input_tensor(waveform);

  // Create SoxEffectsChain
  const auto dtype = waveform.dtype();
  torchaudio::sox_effects_chain::SoxEffectsChain chain(
      /*input_encoding=*/get_tensor_encodinginfo(dtype),
      /*output_encoding=*/get_tensor_encodinginfo(dtype));

  // Prepare output buffer
  std::vector<sox_sample_t> out_buffer;
  out_buffer.reserve(waveform.numel());

  // Build and run effects chain
  chain.addInputTensor(&waveform, sample_rate, channels_first);
  for (const auto& effect : effects) {
    chain.addEffect(effect);
  }
  chain.addOutputBuffer(&out_buffer);
  chain.run();

  // Create tensor from buffer
  auto out_tensor = convert_to_tensor(
      /*buffer=*/out_buffer.data(),
      /*num_samples=*/out_buffer.size(),
      /*num_channels=*/chain.getOutputNumChannels(),
      dtype,
      /*noramlize=*/false,
      channels_first);

  return std::tuple<torch::Tensor, int64_t>(
      out_tensor, chain.getOutputSampleRate());
}

std::tuple<torch::Tensor, int64_t> apply_effects_file(
    const std::string path,
    std::vector<std::vector<std::string>> effects,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
    c10::optional<std::string>& format) {
  // Open input file
  SoxFormat sf(sox_open_read(
      path.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/format.has_value() ? format.value().c_str() : nullptr));

  validate_input_file(sf);

  const auto dtype = get_dtype(sf->encoding.encoding, sf->signal.precision);

  // Prepare output
  std::vector<sox_sample_t> out_buffer;
  out_buffer.reserve(sf->signal.length);

  // Create and run SoxEffectsChain
  torchaudio::sox_effects_chain::SoxEffectsChain chain(
      /*input_encoding=*/sf->encoding,
      /*output_encoding=*/get_tensor_encodinginfo(dtype));

  chain.addInputFile(sf);
  for (const auto& effect : effects) {
    chain.addEffect(effect);
  }
  chain.addOutputBuffer(&out_buffer);
  chain.run();

  // Create tensor from buffer
  bool channels_first_ = channels_first.value_or(true);
  auto tensor = convert_to_tensor(
      /*buffer=*/out_buffer.data(),
      /*num_samples=*/out_buffer.size(),
      /*num_channels=*/chain.getOutputNumChannels(),
      dtype,
      normalize.value_or(true),
      channels_first_);

  return std::tuple<torch::Tensor, int64_t>(
      tensor, chain.getOutputSampleRate());
}

#ifdef TORCH_API_INCLUDE_EXTENSION_H

// Streaming decoding over file-like object is tricky because libsox operates on
// FILE pointer. The folloing is what `sox` and `play` commands do
//  - file input -> FILE pointer
//  - URL input -> call wget in suprocess and pipe the data -> FILE pointer
//  - stdin -> FILE pointer
//
// We want to, instead, fetch byte strings chunk by chunk, consume them, and
// discard.
//
// Here is the approach
// 1. Initialize sox_format_t using sox_open_mem_read, providing the initial
// chunk of byte string
//    This will perform header-based format detection, if necessary, then fill
//    the metadata of sox_format_t. Internally, sox_open_mem_read uses fmemopen,
//    which returns FILE* which points the buffer of the provided byte string.
// 2. Each time sox reads a chunk from the FILE*, we update the underlying
// buffer in a way that it
//    starts with unseen data, and append the new data read from the given
//    fileobj. This will trick libsox as if it keeps reading from the FILE*
//    continuously.
// For Step 2. see `fileobj_input_drain` function in effects_chain.cpp
std::tuple<torch::Tensor, int64_t> apply_effects_fileobj(
    py::object fileobj,
    std::vector<std::vector<std::string>> effects,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
    c10::optional<std::string>& format) {
  // Prepare the buffer used throughout the lifecycle of SoxEffectChain.
  //
  // For certain format (such as FLAC), libsox keeps reading the content at
  // the initialization unless it reaches EOF even when the header is properly
  // parsed. (Making buffer size 8192, which is way bigger than the header,
  // resulted in libsox consuming all the buffer content at the time it opens
  // the file.) Therefore buffer has to always contain valid data, except after
  // EOF. We default to `sox_get_globals()->bufsiz`* for buffer size and we
  // first check if there is enough data to fill the buffer. `read_fileobj`
  // repeatedly calls `read`  method until it receives the requested length of
  // bytes or it reaches EOF. If we get bytes shorter than requested, that means
  // the whole audio data are fetched.
  //
  // * This can be changed with `torchaudio.utils.sox_utils.set_buffer_size`.
  auto capacity =
      (sox_get_globals()->bufsiz > 256) ? sox_get_globals()->bufsiz : 256;
  std::string buffer(capacity, '\0');
  auto* in_buf = const_cast<char*>(buffer.data());
  auto num_read = read_fileobj(&fileobj, capacity, in_buf);
  // If the file is shorter than 256, then libsox cannot read the header.
  auto in_buffer_size = (num_read > 256) ? num_read : 256;

  // Open file (this starts reading the header)
  // When opening a file there are two functions that can touches FILE*.
  // * `auto_detect_format`
  //   https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/formats.c#L43
  // * `startread` handler of detected format.
  //   https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/formats.c#L574
  // To see the handler of a particular format, go to
  //   https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/<FORMAT>.c
  // For example, voribs can be found
  //   https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/vorbis.c#L97-L158
  SoxFormat sf(sox_open_mem_read(
      in_buf,
      in_buffer_size,
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/format.has_value() ? format.value().c_str() : nullptr));

  // In case of streamed data, length can be 0
  validate_input_file(sf, /*check_length=*/false);

  // Prepare output buffer
  std::vector<sox_sample_t> out_buffer;
  out_buffer.reserve(sf->signal.length);

  // Create and run SoxEffectsChain
  const auto dtype = get_dtype(sf->encoding.encoding, sf->signal.precision);
  torchaudio::sox_effects_chain::SoxEffectsChain chain(
      /*input_encoding=*/sf->encoding,
      /*output_encoding=*/get_tensor_encodinginfo(dtype));
  chain.addInputFileObj(sf, in_buf, in_buffer_size, &fileobj);
  for (const auto& effect : effects) {
    chain.addEffect(effect);
  }
  chain.addOutputBuffer(&out_buffer);
  chain.run();

  // Create tensor from buffer
  bool channels_first_ = channels_first.value_or(true);
  auto tensor = convert_to_tensor(
      /*buffer=*/out_buffer.data(),
      /*num_samples=*/out_buffer.size(),
      /*num_channels=*/chain.getOutputNumChannels(),
      dtype,
      normalize.value_or(true),
      channels_first_);

  return std::make_tuple(
      tensor, static_cast<int64_t>(chain.getOutputSampleRate()));
}

#endif // TORCH_API_INCLUDE_EXTENSION_H

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "torchaudio::sox_effects_initialize_sox_effects",
      &torchaudio::sox_effects::initialize_sox_effects);
  m.def(
      "torchaudio::sox_effects_shutdown_sox_effects",
      &torchaudio::sox_effects::shutdown_sox_effects);
  m.def(
      "torchaudio::sox_effects_apply_effects_tensor",
      &torchaudio::sox_effects::apply_effects_tensor);
  m.def(
      "torchaudio::sox_effects_apply_effects_file",
      &torchaudio::sox_effects::apply_effects_file);
}

} // namespace sox_effects
} // namespace torchaudio
