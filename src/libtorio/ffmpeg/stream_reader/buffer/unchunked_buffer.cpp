#include <libtorio/ffmpeg/stream_reader/buffer/unchunked_buffer.h>

namespace torio::io::detail {

UnchunkedBuffer::UnchunkedBuffer(AVRational time_base) : time_base(time_base){};

bool UnchunkedBuffer::is_ready() const {
  return chunks.size() > 0;
}

void UnchunkedBuffer::push_frame(torch::Tensor frame, int64_t pts_) {
  if (chunks.size() == 0) {
    pts = double(pts_) * time_base.num / time_base.den;
  }
  chunks.push_back(frame);
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

} // namespace torio::io::detail
