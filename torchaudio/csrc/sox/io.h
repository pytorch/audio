#ifndef TORCHAUDIO_SOX_IO_H
#define TORCHAUDIO_SOX_IO_H

#include <torch/script.h>
#include <torchaudio/csrc/sox/utils.h>

namespace torchaudio::sox {

auto get_effects(
    const c10::optional<int64_t>& frame_offset,
    const c10::optional<int64_t>& num_frames)
    -> std::vector<std::vector<std::string>>;

using MetaDataTuple =
    std::tuple<int64_t, int64_t, int64_t, int64_t, std::string>;

c10::optional<MetaDataTuple> get_info_file(
    const std::string& path,
    const c10::optional<std::string>& format);

c10::optional<std::tuple<torch::Tensor, int64_t>> load_audio_file(
    const std::string& path,
    const c10::optional<int64_t>& frame_offset,
    const c10::optional<int64_t>& num_frames,
    c10::optional<bool> normalize,
    c10::optional<bool> channels_first,
    const c10::optional<std::string>& format);

void save_audio_file(
    const std::string& path,
    torch::Tensor tensor,
    int64_t sample_rate,
    bool channels_first,
    c10::optional<double> compression,
    c10::optional<std::string> format,
    c10::optional<std::string> encoding,
    c10::optional<int64_t> bits_per_sample);

} // namespace torchaudio::sox

#endif
