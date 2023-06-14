#pragma once
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/typedefs.h>

namespace torchaudio::io::detail {

class ChunkedBuffer {
  // Each AVFrame is converted to a Tensor and stored here.
  std::deque<torch::Tensor> chunks;
  // Time stamps corresponding the first frame of each chunk
  std::deque<int64_t> pts;
  AVRational time_base;

  // The number of frames to return as a chunk
  // If <0, then user wants to receive all the frames
  const int64_t frames_per_chunk;
  // The numbe of chunks to retain
  const int64_t num_chunks;
  // The number of currently stored chunks
  // For video, one Tensor corresponds to one frame, but for audio,
  // one Tensor contains multiple samples, so we track here.
  int64_t num_buffered_frames = 0;

 public:
  ChunkedBuffer(AVRational time_base, int frames_per_chunk, int num_chunks);

  bool is_ready() const;
  void flush();
  c10::optional<Chunk> pop_chunk();
  void push_frame(torch::Tensor frame, int64_t pts_);
};

} // namespace torchaudio::io::detail
