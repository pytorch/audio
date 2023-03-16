#pragma once
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer.h>

namespace torchaudio::io::detail {

//////////////////////////////////////////////////////////////////////////////
// Chunked Buffer Implementation
//////////////////////////////////////////////////////////////////////////////
// Common to both audio and video
template <typename Converter>
class ChunkedBuffer : public Buffer {
  // Each AVFrame is converted to a Tensor and stored here.
  std::deque<torch::Tensor> chunks;
  // Time stamps corresponding the first frame of each chunk
  std::deque<double> pts;
  // Duration of one frame, used to recalculate the PTS of audio samples
  double frame_duration;

  // The number of frames to return as a chunk
  // If <0, then user wants to receive all the frames
  const int64_t frames_per_chunk;
  // The numbe of chunks to retain
  const int64_t num_chunks;
  // The number of currently stored chunks
  // For video, one Tensor corresponds to one frame, but for audio,
  // one Tensor contains multiple samples, so we track here.
  int64_t num_buffered_frames = 0;

  Converter converter;

 public:
  ChunkedBuffer(
      int frames_per_chunk,
      int num_chunks,
      double frame_duration,
      Converter&& converter);

  bool is_ready() const override;
  void flush() override;
  c10::optional<Chunk> pop_chunk() override;
  void push_frame(AVFrame* frame_, double pts_) override;
};

std::unique_ptr<Buffer> get_chunked_buffer(
    int frames_per_chunk,
    int num_chunks,
    double frame_duration,
    AVSampleFormat fmt,
    int num_channels);

std::unique_ptr<Buffer> get_chunked_buffer(
    int frames_per_chunk,
    int num_chunks,
    double frame_duration,
    AVPixelFormat fmt,
    int height,
    int width,
    const torch::Device& device);

} // namespace torchaudio::io::detail
