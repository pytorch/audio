#pragma once
#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace io {
namespace detail {

//////////////////////////////////////////////////////////////////////////////
// Helper functions
//////////////////////////////////////////////////////////////////////////////
torch::Tensor convert_audio(AVFrame* frame);

torch::Tensor convert_image(AVFrame* frame, const torch::Device& device);

} // namespace detail
} // namespace io
} // namespace torchaudio
