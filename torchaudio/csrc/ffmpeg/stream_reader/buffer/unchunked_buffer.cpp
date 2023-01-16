#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/common.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/unchunked_buffer.h>

namespace torchaudio {
namespace ffmpeg {

UnchunkedVideoBuffer::UnchunkedVideoBuffer(const torch::Device& device)
    : device(device) {}

bool UnchunkedBuffer::is_ready() const {
  return chunks.size() > 0;
}

void UnchunkedBuffer::push_tensor(const torch::Tensor& t) {
  chunks.push_back(t);
}

void UnchunkedAudioBuffer::push_frame(AVFrame* frame) {
  push_tensor(detail::convert_audio(frame));
}

void UnchunkedVideoBuffer::push_frame(AVFrame* frame) {
  push_tensor(detail::convert_image(frame, device));
}

c10::optional<torch::Tensor> UnchunkedBuffer::pop_chunk() {
  if (chunks.size() == 0) {
    return {};
  }

  auto ret =
      torch::cat(std::vector<torch::Tensor>{chunks.begin(), chunks.end()}, 0);
  chunks.clear();
  return {ret};
}

void UnchunkedBuffer::flush() {
  chunks.clear();
}

} // namespace ffmpeg
} // namespace torchaudio
