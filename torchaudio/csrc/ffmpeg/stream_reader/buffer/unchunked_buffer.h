#pragma once
#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer.h>
#include <deque>

namespace torchaudio {
namespace ffmpeg {
namespace detail {

//////////////////////////////////////////////////////////////////////////////
// Unchunked Buffer Interface
//////////////////////////////////////////////////////////////////////////////
// Partial implementation for unchunked buffer common to both audio and video
// Used for buffering audio/video streams without chunking
class UnchunkedBuffer : public Buffer {
  // Each AVFrame is converted to a Tensor and stored here.
  std::deque<torch::Tensor> chunks;
  double pts = -1.;

 protected:
  void push_tensor(const torch::Tensor& t, double pts);

 public:
  bool is_ready() const override;
  c10::optional<Chunk> pop_chunk() override;
  void flush() override;
};

class UnchunkedAudioBuffer : public UnchunkedBuffer {
 public:
  void push_frame(AVFrame* frame, double pts) override;
};

class UnchunkedVideoBuffer : public UnchunkedBuffer {
  const torch::Device device;

 public:
  explicit UnchunkedVideoBuffer(const torch::Device& device);

  void push_frame(AVFrame* frame, double pts) override;
};

} // namespace detail
} // namespace ffmpeg
} // namespace torchaudio
