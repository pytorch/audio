#ifndef TORCHAUDIO_SOX_IO_H
#define TORCHAUDIO_SOX_IO_H

#include <torch/script.h>
#include <torchaudio/csrc/sox_utils.h>
#include <torchaudio/csrc/typedefs.h>

namespace torchaudio {
namespace sox_io {

c10::intrusive_ptr<torchaudio::SignalInfo> get_info(const std::string& path);

c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal> load_audio_file(
    const std::string& path,
    const int64_t frame_offset = 0,
    const int64_t num_frames = -1,
    const bool normalize = true,
    const bool channels_first = true);

} // namespace sox_io
} // namespace torchaudio

#endif
