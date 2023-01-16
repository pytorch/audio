#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/chunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/common.h>

namespace torchaudio {
namespace ffmpeg {
namespace detail {

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

void ChunkedBuffer::push_tensor(torch::Tensor frame) {
  using namespace torch::indexing;
  // Note:
  // Audio tensors contain multiple frames while video tensors contain only
  // one frame. Video tensors can be regarded as special degenerated case of
  // audio, so in the following, we only consider audio processing.
  //
  // The incoming Tensor might contain more frames than the value of
  // `frames_per_chunk`.
  // If we push the input tensor to dequeu as-is, then, at the trimming stage,
  // the entire frames would be trimmed, this is not ideal. We want to keep
  // at most `frames_per_chunk * num_chunks` frames.
  // So we slice push the incoming Tensor.
  //

  // 1. Check if the last chunk is fully filled. If not, fill it.
  //
  //  <----- frames per chunk ----->^
  //  x x x x x x x x x x x x x x x |
  //  x x x x x x x + + + + + + - - | num_chunks
  //  - - - - - - - - - - - - - - - |
  //  <-- filled --><--- remain --->v
  //                <- append->
  //
  if (int64_t filled = num_buffered_frames % frames_per_chunk) {
    int64_t num_frames = frame.size(0);
    int64_t remain = frames_per_chunk - filled;
    int64_t append = remain < num_frames ? remain : num_frames;

    torch::Tensor prev = chunks.back();
    // prev[filled:filled+append] = frame[:append]
    prev.index_put_(
        {Slice(filled, filled + append)}, frame.index({Slice(None, append)}));
    num_buffered_frames += append;
    // frame = frame[append:]
    frame = frame.index({Slice(append)});
  }

  // 2. Return if the number of input frames are smaller than the empty buffer.
  // i.e. all the frames are pushed.
  if (frame.numel() == 0) {
    return;
  }

  // 3. Now the existing buffer chunks are fully filled, start adding new chunks
  //
  //  <----- frames per chunk ----->^
  //  x x x x x x x x x x x x x x x |
  //  x x x x x x x x x x x x x x x | num_chunks
  //  + + + + + + + + + + + + + + + |
  //  <---------- append ---------->v
  //
  int64_t num_frames = frame.size(0);
  int64_t num_splits =
      num_frames / frames_per_chunk + (num_frames % frames_per_chunk ? 1 : 0);
  for (int64_t i = 0; i < num_splits; ++i) {
    int64_t start = i * frames_per_chunk;
    // chunk = frame[i*frames_per_chunk:(i+1) * frames_per_chunk]
    auto chunk = frame.index({Slice(start, start + frames_per_chunk)});
    int64_t chunk_size = chunk.size(0);
    TORCH_INTERNAL_ASSERT(
        chunk_size <= frames_per_chunk,
        "Chunk size is larger than frames per chunk. Please file an issue.");
    if (chunk_size < frames_per_chunk) {
      auto shape = chunk.sizes().vec();
      shape[0] = frames_per_chunk;
      auto temp = torch::empty(shape, frame.options());
      temp.index_put_({Slice(None, chunk_size)}, chunk);
      chunk = temp;
    }
    chunks.push_back(chunk);
    num_buffered_frames += chunk_size;

    // Trim if num_chunks > 0
    if (num_chunks > 0 && chunks.size() > num_chunks) {
      TORCH_WARN_ONCE(
          "The number of buffered frames exceeded the buffer size. "
          "Dropping the old frames. "
          "To avoid this, you can set a higher buffer_chunk_size value.");
      chunks.pop_front();
      num_buffered_frames -= frames_per_chunk;
    }
  }
}

c10::optional<torch::Tensor> ChunkedBuffer::pop_chunk() {
  using namespace torch::indexing;
  if (!num_buffered_frames) {
    return {};
  }
  torch::Tensor ret = chunks.front();
  chunks.pop_front();
  if (num_buffered_frames < frames_per_chunk) {
    ret = ret.index({Slice(None, num_buffered_frames)});
  }
  num_buffered_frames -= ret.size(0);
  return c10::optional<torch::Tensor>{ret};
}

void ChunkedAudioBuffer::push_frame(AVFrame* frame) {
  push_tensor(convert_audio(frame));
}

void ChunkedVideoBuffer::push_frame(AVFrame* frame) {
  push_tensor(convert_image(frame, device));
}

void ChunkedBuffer::flush() {
  chunks.clear();
}

} // namespace detail
} // namespace ffmpeg
} // namespace torchaudio
