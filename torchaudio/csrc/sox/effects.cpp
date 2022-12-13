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
      TORCH_CHECK(
          sox_init() == SOX_SUCCESS, "Failed to initialize sox effects.");
      SOX_RESOURCE_STATE = Initialized;
      break;
    case Initialized:
      break;
    case ShutDown:
      TORCH_CHECK(
          false, "SoX Effects has been shut down. Cannot initialize again.");
  }
};

void shutdown_sox_effects() {
  const std::lock_guard<std::mutex> lock(SOX_RESOUCE_STATE_MUTEX);

  switch (SOX_RESOURCE_STATE) {
    case NotInitialized:
      TORCH_CHECK(false, "SoX Effects is not initialized. Cannot shutdown.");
    case Initialized:
      TORCH_CHECK(
          sox_quit() == SOX_SUCCESS, "Failed to initialize sox effects.");
      SOX_RESOURCE_STATE = ShutDown;
      break;
    case ShutDown:
      break;
  }
}

auto apply_effects_tensor(
    torch::Tensor waveform,
    int64_t sample_rate,
    const std::vector<std::vector<std::string>>& effects,
    bool channels_first) -> std::tuple<torch::Tensor, int64_t> {
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
      /*normalize=*/false,
      channels_first);

  return std::tuple<torch::Tensor, int64_t>(
      out_tensor, chain.getOutputSampleRate());
}

auto apply_effects_file(
    const std::string& path,
    const std::vector<std::vector<std::string>>& effects,
    c10::optional<bool> normalize,
    c10::optional<bool> channels_first,
    const c10::optional<std::string>& format)
    -> c10::optional<std::tuple<torch::Tensor, int64_t>> {
  // Open input file
  SoxFormat sf(sox_open_read(
      path.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/format.has_value() ? format.value().c_str() : nullptr));

  if (static_cast<sox_format_t*>(sf) == nullptr ||
      sf->encoding.encoding == SOX_ENCODING_UNKNOWN) {
    return {};
  }

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
