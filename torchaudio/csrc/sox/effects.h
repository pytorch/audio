#ifndef TORCHAUDIO_SOX_EFFECTS_H
#define TORCHAUDIO_SOX_EFFECTS_H

#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>
#endif // TORCH_API_INCLUDE_EXTENSION_H

#include <torch/script.h>
#include <torchaudio/csrc/sox/utils.h>

namespace torchaudio {
namespace sox_effects {

void initialize_sox_effects();

void shutdown_sox_effects();

std::tuple<torch::Tensor, int64_t> apply_effects_tensor(
    torch::Tensor waveform,
    int64_t sample_rate,
    std::vector<std::vector<std::string>> effects,
    bool channels_first);

std::tuple<torch::Tensor, int64_t> apply_effects_file(
    const std::string path,
    std::vector<std::vector<std::string>> effects,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
    c10::optional<std::string>& format);

#ifdef TORCH_API_INCLUDE_EXTENSION_H

std::tuple<torch::Tensor, int64_t> apply_effects_fileobj(
    py::object fileobj,
    std::vector<std::vector<std::string>> effects,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
    c10::optional<std::string>& format);

#endif // TORCH_API_INCLUDE_EXTENSION_H

} // namespace sox_effects
} // namespace torchaudio

#endif
