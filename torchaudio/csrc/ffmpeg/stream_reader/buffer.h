#pragma once
#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <deque>

namespace torchaudio {
namespace ffmpeg {

class Buffer {
 protected:
  // Each AVFrame is converted to a Tensor and stored here.
  std::deque<torch::Tensor> chunks;

  // The number of frames to return as a chunk
  // If <0, then user wants to receive all the frames
  const int frames_per_chunk;
  // The numbe of chunks to retain
  const int num_chunks;
  // The number of currently stored chunks
  // For video, one Tensor corresponds to one frame, but for audio,
  // one Tensor contains multiple samples, so we track here.
  int num_buffered_frames = 0;

 public:
  Buffer(int frames_per_chunk, int num_chunks);
  virtual ~Buffer() = default;

  //////////////////////////////////////////////////////////////////////////////
  // Query
  //////////////////////////////////////////////////////////////////////////////
  // Check if buffeer has enoough number of frames for a chunk
  // If frame_per_chunk <0, returns true if there is >0 frames.
  // Otherwise, returns if num_frames >= frame_per_chunk.
  bool is_ready() const;

  //////////////////////////////////////////////////////////////////////////////
  // Modifiers
  //////////////////////////////////////////////////////////////////////////////
  virtual void push_frame(AVFrame* frame) = 0;

  c10::optional<torch::Tensor> pop_chunk();

  void flush();

 private:
  virtual torch::Tensor pop_one_chunk() = 0;
  torch::Tensor pop_all();
};

// Specialization of the handling around push/pop for audio/video.

////////////////////////////////////////////////////////////////////////////////
// AudioBuffer specialization
////////////////////////////////////////////////////////////////////////////////
// For audio, input AVFrame contains multiple frames.
// When popping the buffered frames chunk-by-chunk, it is easier if they are
// organized by chunk when pushed to deque object.
// Therefore, audio implements pushing mechanism that makes sure that
// each Tensor in deque consists Tensors with `frames_per_chunk` frames.
class AudioBuffer : public Buffer {
 public:
  AudioBuffer(int frames_per_chunk, int num_chunks);

  void push_frame(AVFrame* frame);

 private:
  void push_tensor(torch::Tensor tensor);
  torch::Tensor pop_one_chunk();
};

////////////////////////////////////////////////////////////////////////////////
// VideoBuffer specialization
////////////////////////////////////////////////////////////////////////////////
// For video, input AVFrame contains one frame.
// Contraty to audio, it is simple to push one frame each time to deque.
// But this mean that chunks consisting of multiple frames have to be created
// at popping time.
class VideoBuffer : public Buffer {
  const torch::Device device;

 public:
  VideoBuffer(
      int frames_per_chunk,
      int num_chunks,
      const torch::Device& device);

  void push_frame(AVFrame* frame);

 private:
  void push_tensor(torch::Tensor tensor);
  torch::Tensor pop_one_chunk();
};

} // namespace ffmpeg
} // namespace torchaudio
