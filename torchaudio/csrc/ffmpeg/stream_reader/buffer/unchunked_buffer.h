#pragma once
#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/typedefs.h>
#include <deque>

namespace torchaudio::io::detail {

class UnchunkedBuffer {
  // Each AVFrame is converted to a Tensor and stored here.
  std::deque<torch::Tensor> chunks;
  double pts = -1.;
  AVRational time_base;

 public:
  UnchunkedBuffer(AVRational time_base);
  bool is_ready() const;
  void push_frame(torch::Tensor frame, int64_t pts_);
  c10::optional<Chunk> pop_chunk();
  void flush();
};

} // namespace torchaudio::io::detail
