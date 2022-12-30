#pragma once
#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace ffmpeg {
namespace detail {

struct ImageBuffer {
  torch::Tensor buffer;
  bool is_planar;
};

torch::Tensor convert_audio(AVFrame* frame);

ImageBuffer get_image_buffer(
    AVFrame* pFrame,
    int num_frames,
    const torch::Device& device);

void write_image(AVFrame* frame, torch::Tensor& buf);

} // namespace detail
} // namespace ffmpeg
} // namespace torchaudio
