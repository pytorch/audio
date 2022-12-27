#pragma once
#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer.h>
#include <deque>

namespace torchaudio {
namespace ffmpeg {

//////////////////////////////////////////////////////////////////////////////
// Unchunked Buffer Interface
//////////////////////////////////////////////////////////////////////////////
// Partial implementation for unchunked buffer common to both audio and video
// Used for buffering audio/video streams without chunking
class UnchunkedBuffer : public Buffer {
  // Each AVFrame is converted to a Tensor and stored here.
  std::deque<torch::Tensor> chunks;

 protected:
  // The number of currently stored chunks
  // For video, one Tensor corresponds to one frame, but for audio,
  // one Tensor contains multiple samples, so we track here.
  int64_t num_buffered_frames = 0;
  void push_tensor(const torch::Tensor& t);

 public:
  bool is_ready() const override;
  c10::optional<torch::Tensor> pop_chunk() override;
  void flush() override;
};

class UnchunkedAudioBuffer : public UnchunkedBuffer {
 public:
  void push_frame(AVFrame* frame) override;
};

class UnchunkedVideoBuffer : public UnchunkedBuffer {
  const torch::Device device;

 public:
  explicit UnchunkedVideoBuffer(const torch::Device& device);

  void push_frame(AVFrame* frame) override;
};

} // namespace ffmpeg
} // namespace torchaudio
