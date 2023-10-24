#pragma once
#include <libtorio/ffmpeg/ffmpeg.h>
#include <libtorio/ffmpeg/stream_reader/typedefs.h>
#include <torch/types.h>
#include <deque>

namespace torio::io::detail {

class UnchunkedBuffer {
  // Each AVFrame is converted to a Tensor and stored here.
  std::deque<torch::Tensor> chunks;
  double pts = -1.;
  AVRational time_base;

 public:
  explicit UnchunkedBuffer(AVRational time_base);
  bool is_ready() const;
  void push_frame(torch::Tensor frame, int64_t pts_);
  c10::optional<Chunk> pop_chunk();
  void flush();
};

} // namespace torio::io::detail
