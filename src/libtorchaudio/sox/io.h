#ifndef TORCHAUDIO_SOX_IO_H
#define TORCHAUDIO_SOX_IO_H

#include <libtorchaudio/sox/utils.h>
#include <torch/script.h>

namespace torchaudio::sox {

auto get_effects(
    const std::optional<int64_t>& frame_offset,
    const std::optional<int64_t>& num_frames)
    -> std::vector<std::vector<std::string>>;

std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> get_info_file(
    const std::string& path,
    const std::optional<std::string>& format);

std::tuple<torch::Tensor, int64_t> load_audio_file(
    const std::string& path,
    const std::optional<int64_t>& frame_offset,
    const std::optional<int64_t>& num_frames,
    std::optional<bool> normalize,
    std::optional<bool> channels_first,
    const std::optional<std::string>& format);

void save_audio_file(
    const std::string& path,
    torch::Tensor tensor,
    int64_t sample_rate,
    bool channels_first,
    std::optional<double> compression,
    std::optional<std::string> format,
    std::optional<std::string> encoding,
    std::optional<int64_t> bits_per_sample);

} // namespace torchaudio::sox

#endif
