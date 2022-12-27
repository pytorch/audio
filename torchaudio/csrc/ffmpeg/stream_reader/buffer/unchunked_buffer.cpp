#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/unchunked_buffer.h>

namespace torchaudio {
namespace ffmpeg {

UnchunkedVideoBuffer::UnchunkedVideoBuffer(const torch::Device& device)
    : device(device) {}

bool UnchunkedBuffer::is_ready() const {
  return num_buffered_frames > 0;
}

void UnchunkedBuffer::push_tensor(const torch::Tensor& t) {
  // If frames_per_chunk < 0, users want to fetch all frames.
  // Just push back to chunks and that's it.
  chunks.push_back(t);
  num_buffered_frames += t.size(0);
}

void UnchunkedAudioBuffer::push_frame(AVFrame* frame) {
  push_tensor(convert_audio_tensor(frame));
}

void UnchunkedVideoBuffer::push_frame(AVFrame* frame) {
  push_tensor(convert_image_tensor(frame, device));
}

c10::optional<torch::Tensor> UnchunkedBuffer::pop_chunk() {
  if (!num_buffered_frames) {
    return c10::optional<torch::Tensor>{};
  }

  std::vector<torch::Tensor> ret;
  while (chunks.size()) {
    torch::Tensor& t = chunks.front();
    int64_t n_frames = t.size(0);
    ret.push_back(t);
    chunks.pop_front();
    num_buffered_frames -= n_frames;
  }
  return c10::optional<torch::Tensor>{torch::cat(ret, 0)};
}

void UnchunkedBuffer::flush() {
  chunks.clear();
}

} // namespace ffmpeg
} // namespace torchaudio
