#ifndef TORCHAUDIO_SOX_EFFECTS_H
#define TORCHAUDIO_SOX_EFFECTS_H

#include <libtorchaudio/sox/utils.h>
#include <torch/script.h>

namespace torchaudio::sox {

void initialize_sox_effects();

void shutdown_sox_effects();

auto apply_effects_tensor(
    torch::Tensor waveform,
    int64_t sample_rate,
    const std::vector<std::vector<std::string>>& effects,
    bool channels_first) -> std::tuple<torch::Tensor, int64_t>;

auto apply_effects_file(
    const std::string& path,
    const std::vector<std::vector<std::string>>& effects,
    std::optional<bool> normalize,
    std::optional<bool> channels_first,
    const std::optional<std::string>& format)
    -> std::tuple<torch::Tensor, int64_t>;

} // namespace torchaudio::sox

#endif
