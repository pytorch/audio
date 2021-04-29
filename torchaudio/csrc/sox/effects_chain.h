#ifndef TORCHAUDIO_SOX_EFFECTS_CHAIN_H
#define TORCHAUDIO_SOX_EFFECTS_CHAIN_H

#include <sox.h>
#include <torchaudio/csrc/sox/utils.h>

#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>
#endif // TORCH_API_INCLUDE_EXTENSION_H

namespace torchaudio {
namespace sox_effects_chain {

// Helper struct to safely close sox_effects_chain_t with handy methods
class SoxEffectsChain {
  const sox_encodinginfo_t in_enc_;
  const sox_encodinginfo_t out_enc_;
  sox_signalinfo_t in_sig_;
  sox_signalinfo_t interm_sig_;
  sox_signalinfo_t out_sig_;
  sox_effects_chain_t* sec_;

 public:
  explicit SoxEffectsChain(
      sox_encodinginfo_t input_encoding,
      sox_encodinginfo_t output_encoding);
  SoxEffectsChain(const SoxEffectsChain& other) = delete;
  SoxEffectsChain(const SoxEffectsChain&& other) = delete;
  SoxEffectsChain& operator=(const SoxEffectsChain& other) = delete;
  SoxEffectsChain& operator=(SoxEffectsChain&& other) = delete;
  ~SoxEffectsChain();
  void run();
  void addInputTensor(
      torch::Tensor* waveform,
      int64_t sample_rate,
      bool channels_first);
  void addInputFile(sox_format_t* sf);
  void addOutputBuffer(std::vector<sox_sample_t>* output_buffer);
  void addOutputFile(sox_format_t* sf);
  void addEffect(const std::vector<std::string> effect);
  int64_t getOutputNumChannels();
  int64_t getOutputSampleRate();

#ifdef TORCH_API_INCLUDE_EXTENSION_H

  void addInputFileObj(
      sox_format_t* sf,
      char* buffer,
      uint64_t buffer_size,
      py::object* fileobj);

  void addOutputFileObj(
      sox_format_t* sf,
      char** buffer,
      size_t* buffer_size,
      py::object* fileobj);

#endif // TORCH_API_INCLUDE_EXTENSION_H
};

} // namespace sox_effects_chain
} // namespace torchaudio

#endif
