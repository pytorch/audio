#include <sox.h>
#include <torchaudio/csrc/sox_effects.h>
#include <torchaudio/csrc/sox_effects_chain.h>
#include <torchaudio/csrc/sox_utils.h>

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

c10::intrusive_ptr<TensorSignal> apply_effects_tensor(
    const c10::intrusive_ptr<TensorSignal>& input_signal,
    std::vector<std::vector<std::string>> effects) {
  auto in_tensor = input_signal->getTensor();
  validate_input_tensor(in_tensor);

  // Create SoxEffectsChain
  const auto dtype = in_tensor.dtype();
  torchaudio::sox_effects_chain::SoxEffectsChain chain(
      /*input_encoding=*/get_encodinginfo("wav", dtype, 0.),
      /*output_encoding=*/get_encodinginfo("wav", dtype, 0.));

  // Prepare output buffer
  std::vector<sox_sample_t> out_buffer;
  out_buffer.reserve(in_tensor.numel());

  // Build and run effects chain
  chain.addInputTensor(input_signal.get());
  for (const auto& effect : effects) {
    chain.addEffect(effect);
  }
  chain.addOutputBuffer(&out_buffer);
  chain.run();

  // Create tensor from buffer
  const auto channels_first = input_signal->getChannelsFirst();
  auto out_tensor = convert_to_tensor(
      /*buffer=*/out_buffer.data(),
      /*num_samples=*/out_buffer.size(),
      /*num_channels=*/chain.getOutputNumChannels(),
      dtype,
      /*noramlize=*/false,
      channels_first);

  return c10::make_intrusive<TensorSignal>(
      out_tensor, chain.getOutputSampleRate(), channels_first);
}

namespace {

c10::intrusive_ptr<TensorSignal> apply_effects_impl(
    const SoxFormat& sf,
    std::vector<std::vector<std::string>> effects,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first) {

  validate_input_file(sf);

  const auto dtype = get_dtype(sf->encoding.encoding, sf->signal.precision);

  // Prepare output
  std::vector<sox_sample_t> out_buffer;
  out_buffer.reserve(sf->signal.length);

  // Create and run SoxEffectsChain
  torchaudio::sox_effects_chain::SoxEffectsChain chain(
      /*input_encoding=*/sf->encoding,
      /*output_encoding=*/get_encodinginfo("wav", dtype, 0.));

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

  return c10::make_intrusive<TensorSignal>(
      tensor, chain.getOutputSampleRate(), channels_first_);
}

} // namespace

c10::intrusive_ptr<TensorSignal> apply_effects_file(
    const std::string& path,
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
  return apply_effects_impl(sf, effects, normalize, channels_first);
}

c10::intrusive_ptr<TensorSignal> apply_effects_bytes(
    const std::string& bytes,
    std::vector<std::vector<std::string>> effects,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
        c10::optional<std::string>& format) {
  SoxFormat sf(sox_open_mem_read(
      static_cast<void*>(const_cast<char*>(bytes.c_str())),
      bytes.length(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/format.has_value() ? format.value().c_str() : nullptr));
  return apply_effects_impl(sf, effects, normalize, channels_first);
}

#ifdef TORCH_API_INCLUDE_EXTENSION_H

c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal> apply_effects_fileobj(
    py::object fileobj,
    std::vector<std::vector<std::string>> effects,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
    c10::optional<std::string>& format) {

  // Streaming decoding over file-like object is tricky because libsox operates on FILE pointer.
  // The folloing is what `sox` and `play` commands do
  //  - file input -> FILE pointer
  //  - URL input -> call wget in suprocess and pipe the data -> FILE pointer
  //  - stdin -> FILE pointer
  //
  // We want to, instead, consume byte strings chunk by chunk, discarding the consumed chunks,
  //
  // Here is the approach
  // 1. Initialize sox_format_t using sox_open_mem_read, providing the initial chunk of byte string
  //    This will perform header based derection, fill the necessary information of sox_format_t.
  //    Internally, sox_open_mem_read uses fmemopen, which returns FILE pointer which points the
  //    buffer of the provided byte string.
  // 2. When new data come, we update the underlying buffer in a way that it starts with unseen data,
  //    then reset the FILE pointer seek location to the beggining.
  //    This will trick libsox as if it keeps reading from the FILE continuously.

  // Fetch the header.
  // 4096 is minimum size requried by auto_detect_format
  // https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/formats.c#L40-L48
  py::bytes header = fileobj.attr("read")(4096);

  // Initialize the buffer object with header data.
  // Converting py::bytes to std::string copies the memory,
  // https://github.com/pybind/pybind11/blob/79b0e2c05206865fe12a1ef95552e059913c1855/include/pybind11/pytypes.h#L1022-L1028
  // therefore, as long as we keep `buffer_holder` variable until effect chain finishes,
  // it is safe to modify the memory pointed by `buf` manually
  std::string buffer_holder = static_cast<std::string>(header);
  void* buf = static_cast<void*>(const_cast<char*>(buffer_holder.c_str()));
  auto buf_size = buffer_holder.length();
  // The following initializes sox_format_t whose FILE* memeber pointing `buf` as underlying storage
  SoxFormat sf(sox_open_mem_read(
      buf,
      buf_size,
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/format.has_value() ? format.value().c_str() : nullptr));

  fseek ((FILE*)sf->fp, 0, SEEK_SET);

  validate_input_file(sf);

  const auto dtype = get_dtype(sf->encoding.encoding, sf->signal.precision);

  // Prepare output
  std::vector<sox_sample_t> out_buffer;
  out_buffer.reserve(sf->signal.length);

  // Create and run SoxEffectsChain
  torchaudio::sox_effects_chain::SoxEffectsChain chain(
      /*input_encoding=*/sf->encoding,
      /*output_encoding=*/get_encodinginfo("wav", dtype, 0.));

  chain.addInputFileObj(sf, buf, buf_size, &fileobj);
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

  return c10::make_intrusive<TensorSignal>(
      tensor, chain.getOutputSampleRate(), channels_first_);
}

#endif // TORCH_API_INCLUDE_EXTENSION_H

} // namespace sox_effects
} // namespace torchaudio
