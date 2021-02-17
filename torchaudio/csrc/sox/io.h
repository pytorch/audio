#ifndef TORCHAUDIO_SOX_IO_H
#define TORCHAUDIO_SOX_IO_H

#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>
#endif // TORCH_API_INCLUDE_EXTENSION_H

#include <torch/script.h>
#include <torchaudio/csrc/sox/utils.h>

namespace torchaudio {
namespace sox_io {

std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> get_info_file(
    const std::string& path,
    const c10::optional<std::string>& format);

std::tuple<torch::Tensor, int64_t> load_audio_file(
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
    c10::optional<std::string> dtype);

#ifdef TORCH_API_INCLUDE_EXTENSION_H

std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> get_info_fileobj(
    py::object fileobj,
    c10::optional<std::string>& format);

std::tuple<torch::Tensor, int64_t> load_audio_fileobj(
    py::object fileobj,
    c10::optional<int64_t>& frame_offset,
    c10::optional<int64_t>& num_frames,
    c10::optional<bool>& normalize,
    c10::optional<bool>& channels_first,
    c10::optional<std::string>& format);

void save_audio_fileobj(
    py::object fileobj,
    torch::Tensor tensor,
    int64_t sample_rate,
    bool channels_first,
    c10::optional<double> compression,
    std::string filetype,
    c10::optional<std::string> dtype);

#endif // TORCH_API_INCLUDE_EXTENSION_H

} // namespace sox_io
} // namespace torchaudio

#endif
