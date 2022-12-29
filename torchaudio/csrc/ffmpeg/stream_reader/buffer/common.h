#pragma once
#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace ffmpeg {
namespace detail {

//////////////////////////////////////////////////////////////////////////////
// Helper functions
//////////////////////////////////////////////////////////////////////////////
torch::Tensor convert_audio(AVFrame* frame);

torch::Tensor get_interlaced_image_buffer(AVFrame* pFrame);
torch::Tensor get_planar_image_buffer(AVFrame* pFrame);
void write_interlaced_image(AVFrame* pFrame, torch::Tensor& frame);
void write_planar_image(AVFrame* pFrame, torch::Tensor& frame);
torch::Tensor convert_image(AVFrame* frame, const torch::Device& device);

} // namespace detail
} // namespace ffmpeg
} // namespace torchaudio
