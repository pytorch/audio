#ifndef TORCHAUDIO_SOX_EFFECTS_H
#define TORCHAUDIO_SOX_EFFECTS_H

#include <torch/script.h>
#include <torchaudio/csrc/sox_utils.h>

namespace torchaudio {
namespace sox_effects {

void initialize_sox_effects();

void shutdown_sox_effects();

c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal> apply_effects_tensor(
    const c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal>& input_signal,
    const std::vector<std::vector<std::string>>& effects);

c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal> apply_effects_file(
    const std::string& path,
    const std::vector<std::vector<std::string>>& effects,
    const bool normalize,
    const bool channels_first);

} // namespace sox_effects
} // namespace torchaudio

#endif
