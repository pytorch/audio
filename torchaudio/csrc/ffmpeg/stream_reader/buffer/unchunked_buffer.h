#pragma once
#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer.h>
#include <deque>

namespace torchaudio::io::detail {

//////////////////////////////////////////////////////////////////////////////
// Unchunked Buffer Interface
//////////////////////////////////////////////////////////////////////////////
// Partial implementation for unchunked buffer common to both audio and video
// Used for buffering audio/video streams without chunking
template <typename Converter>
class UnchunkedBuffer : public Buffer {
  // Each AVFrame is converted to a Tensor and stored here.
  std::deque<torch::Tensor> chunks;
  double pts = -1.;
  AVRational time_base;
  Converter converter;

 public:
  UnchunkedBuffer(AVRational time_base, Converter&& converter);
  bool is_ready() const override;
  void push_frame(AVFrame* frame) override;
  c10::optional<Chunk> pop_chunk() override;
  void flush() override;
};

std::unique_ptr<Buffer> get_unchunked_buffer(
    AVRational time_base,
    AVSampleFormat fmt,
    int num_channels);

std::unique_ptr<Buffer> get_unchunked_buffer(
    AVRational time_base,
    AVPixelFormat fmt,
    int height,
    int width,
    const torch::Device& device);

} // namespace torchaudio::io::detail
