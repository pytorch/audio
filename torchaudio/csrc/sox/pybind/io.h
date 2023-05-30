#ifndef TORCHAUDIO_PYBIND_SOX_IO_H
#define TORCHAUDIO_PYBIND_SOX_IO_H

#include <torch/extension.h>

namespace torchaudio::sox {

using MetaDataTuple =
    std::tuple<int64_t, int64_t, int64_t, int64_t, std::string>;

auto get_info_fileobj(py::object fileobj, c10::optional<std::string> format)
    -> c10::optional<MetaDataTuple>;

auto load_audio_fileobj(
    py::object fileobj,
    c10::optional<int64_t> frame_offset,
    c10::optional<int64_t> num_frames,
    c10::optional<bool> normalize,
    c10::optional<bool> channels_first,
    c10::optional<std::string> format)
    -> c10::optional<std::tuple<torch::Tensor, int64_t>>;

void save_audio_fileobj(
    py::object fileobj,
    torch::Tensor tensor,
    int64_t sample_rate,
    bool channels_first,
    c10::optional<double> compression,
    c10::optional<std::string> format,
    c10::optional<std::string> encoding,
    c10::optional<int64_t> bits_per_sample);

} // namespace torchaudio::sox

#endif
