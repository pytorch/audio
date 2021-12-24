#pragma once
#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <deque>

namespace torchaudio {
namespace ffmpeg {

class Buffer {
  std::deque<torch::Tensor> chunks;
  AVMediaType media_type;

  void push_audio_frame(AVFrame* pFrame);
  void push_video_frame(AVFrame* pFrame);

 public:
  Buffer(AVMediaType type);

  void push_frame(AVFrame* pFrame);
  torch::Tensor pop_all();
};

} // namespace ffmpeg
} // namespace torchaudio
