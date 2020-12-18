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
    const std::vector<std::vector<std::string>>& effects) {
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

c10::intrusive_ptr<TensorSignal> apply_effects_file(
    const std::string& path,
    const std::vector<std::vector<std::string>>& effects,
    const bool normalize,
    const bool channels_first) {
  // Open input file
  SoxFormat sf(sox_open_read(
      path.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));

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
  auto tensor = convert_to_tensor(
      /*buffer=*/out_buffer.data(),
      /*num_samples=*/out_buffer.size(),
      /*num_channels=*/chain.getOutputNumChannels(),
      dtype,
      normalize,
      channels_first);

  return c10::make_intrusive<TensorSignal>(
      tensor, chain.getOutputSampleRate(), channels_first);
}

} // namespace sox_effects
} // namespace torchaudio
