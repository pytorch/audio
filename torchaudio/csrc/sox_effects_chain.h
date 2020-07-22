#ifndef TORCHAUDIO_SOX_EFFECTS_CHAIN_H
#define TORCHAUDIO_SOX_EFFECTS_CHAIN_H

#include <sox.h>
#include <torch/script.h>
#include <torchaudio/csrc/sox_utils.h>

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
  void addInputTensor(torchaudio::sox_utils::TensorSignal* signal);
  void addInputFile(sox_format_t* sf);
  void addOutputBuffer(std::vector<sox_sample_t>* output_buffer);
  void addOutputFile(sox_format_t* sf);
  void addEffect(const std::vector<std::string> effect);
  int64_t getOutputNumChannels();
  int64_t getOutputSampleRate();
};

} // namespace sox_effects_chain
} // namespace torchaudio

#endif
