#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/common.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/unchunked_buffer.h>

namespace torchaudio {
namespace ffmpeg {
namespace detail {

UnchunkedVideoBuffer::UnchunkedVideoBuffer(const torch::Device& device)
    : device(device) {}

bool UnchunkedBuffer::is_ready() const {
  return chunks.size() > 0;
}

void UnchunkedBuffer::push_tensor(const torch::Tensor& t, double pts_) {
  if (chunks.size() == 0) {
    pts = pts_;
  }
  chunks.push_back(t);
}

void UnchunkedAudioBuffer::push_frame(AVFrame* frame, double pts_) {
  push_tensor(convert_audio(frame), pts_);
}

void UnchunkedVideoBuffer::push_frame(AVFrame* frame, double pts_) {
  push_tensor(convert_image(frame, device), pts_);
}

c10::optional<Chunk> UnchunkedBuffer::pop_chunk() {
  if (chunks.size() == 0) {
    return {};
  }

  auto frames =
      torch::cat(std::vector<torch::Tensor>{chunks.begin(), chunks.end()}, 0);
  chunks.clear();
  return {Chunk{frames, pts}};
}

void UnchunkedBuffer::flush() {
  chunks.clear();
}

} // namespace detail
} // namespace ffmpeg
} // namespace torchaudio
