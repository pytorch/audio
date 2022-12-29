#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/chunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/common.h>

namespace torchaudio {
namespace ffmpeg {

ChunkedBuffer::ChunkedBuffer(int frames_per_chunk, int num_chunks)
    : frames_per_chunk(frames_per_chunk), num_chunks(num_chunks) {}

ChunkedAudioBuffer::ChunkedAudioBuffer(int frames_per_chunk, int num_chunks)
    : ChunkedBuffer(frames_per_chunk, num_chunks) {}

ChunkedVideoBuffer::ChunkedVideoBuffer(
    int frames_per_chunk,
    int num_chunks,
    const torch::Device& device_)
    : ChunkedBuffer(frames_per_chunk, num_chunks), device(device_) {}

bool ChunkedBuffer::is_ready() const {
  return num_buffered_frames >= frames_per_chunk;
}

void ChunkedAudioBuffer::push_tensor(torch::Tensor frame) {
  // Push
  // Note:
  // For audio, the incoming tensor contains multiple of samples.
  // For small `frames_per_chunk` value, it might be more than `max_frames`.
  // If we push the tensor as-is, then, the whole frame might be popped at
  // trimming stage, resulting buffer always empty. So we slice push the
  // incoming Tensor.

  // Check the last inserted Tensor and if the numbe of frames is not
  // frame_per_chunk, reprocess it again with the incomping tensor
  if (num_buffered_frames % frames_per_chunk) {
    torch::Tensor prev = chunks.back();
    chunks.pop_back();
    num_buffered_frames -= prev.size(0);
    frame = torch::cat({prev, frame}, 0);
  }

  while (true) {
    int64_t num_input_frames = frame.size(0);
    if (num_input_frames <= frames_per_chunk) {
      chunks.push_back(frame);
      num_buffered_frames += num_input_frames;
      break;
    }
    // The input tensor contains more frames than frames_per_chunk
    auto splits =
        torch::tensor_split(frame, {frames_per_chunk, num_input_frames});
    chunks.push_back(splits[0]);
    num_buffered_frames += frames_per_chunk;
    frame = splits[1];
  }

  // Trim
  // If frames_per_chunk > 0, we only retain the following number of frames and
  // Discard older frames.
  int64_t max_frames = num_chunks * frames_per_chunk;
  while (num_buffered_frames > max_frames) {
    TORCH_WARN_ONCE(
        "The number of buffered frames exceeded the buffer size. "
        "Dropping the old frames. "
        "To avoid this, you can set a higher buffer_chunk_size value.");
    torch::Tensor& t = chunks.front();
    num_buffered_frames -= t.size(0);
    chunks.pop_front();
  }
}

void ChunkedAudioBuffer::push_frame(AVFrame* frame) {
  push_tensor(detail::convert_audio(frame));
}

void ChunkedVideoBuffer::push_tensor(const torch::Tensor& frame) {
  // the video frames is expected to contain only one frame
  chunks.push_back(frame);
  num_buffered_frames += frame.size(0);

  // Trim
  int64_t max_frames = num_chunks * frames_per_chunk;
  if (num_buffered_frames > max_frames) {
    TORCH_WARN_ONCE(
        "The number of buffered frames exceeded the buffer size. "
        "Dropping the old frames. "
        "To avoid this, you can set a higher buffer_chunk_size value.");
    torch::Tensor& t = chunks.front();
    num_buffered_frames -= t.size(0);
    chunks.pop_front();
  }
}

void ChunkedVideoBuffer::push_frame(AVFrame* frame) {
  push_tensor(detail::convert_image(frame, device));
}

c10::optional<torch::Tensor> ChunkedAudioBuffer::pop_chunk() {
  if (!num_buffered_frames) {
    return {};
  }
  // Audio deque are aligned with `frames_per_chunk`
  torch::Tensor ret = chunks.front();
  chunks.pop_front();
  num_buffered_frames -= ret.size(0);
  return c10::optional<torch::Tensor>{ret};
}

c10::optional<torch::Tensor> ChunkedVideoBuffer::pop_chunk() {
  if (!num_buffered_frames) {
    return {};
  }
  // Video deque contains one frame par one tensor
  std::vector<torch::Tensor> ret;
  while (num_buffered_frames > 0 && ret.size() < frames_per_chunk) {
    torch::Tensor& t = chunks.front();
    ret.push_back(t);
    chunks.pop_front();
    num_buffered_frames -= 1;
  }
  return c10::optional<torch::Tensor>{torch::cat(ret, 0)};
}

void ChunkedBuffer::flush() {
  chunks.clear();
}

} // namespace ffmpeg
} // namespace torchaudio
