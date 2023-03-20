#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/chunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/conversion.h>

namespace torchaudio::io::detail {

template <typename Converter>
ChunkedBuffer<Converter>::ChunkedBuffer(
    AVRational time_base,
    int frames_per_chunk_,
    int num_chunks_,
    Converter&& converter_)
    : time_base(time_base),
      frames_per_chunk(frames_per_chunk_),
      num_chunks(num_chunks_),
      converter(std::move(converter_)){};

template <typename Converter>
bool ChunkedBuffer<Converter>::is_ready() const {
  return num_buffered_frames >= frames_per_chunk;
}

template <typename Converter>
void ChunkedBuffer<Converter>::push_frame(AVFrame* frame_) {
  int64_t pts_ = frame_->pts;
  torch::Tensor frame = converter.convert(frame_);

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
    TORCH_INTERNAL_ASSERT(
        chunks.size() > 0,
        "There is supposed to be left over frames, but the buffer dequeue is empty.");
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
    pts_ += append;
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
    int64_t pts_val = pts_ + start;
    int64_t chunk_size = chunk.size(0);
    TORCH_INTERNAL_ASSERT(
        chunk_size <= frames_per_chunk,
        "Chunk size is larger than frames per chunk.");
    if (chunk_size < frames_per_chunk) {
      auto shape = chunk.sizes().vec();
      shape[0] = frames_per_chunk;
      auto temp = torch::empty(shape, frame.options());
      temp.index_put_({Slice(None, chunk_size)}, chunk);
      chunk = temp;
    }
    chunks.push_back(chunk);
    pts.push_back(pts_val);
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

template <typename Converter>
c10::optional<Chunk> ChunkedBuffer<Converter>::pop_chunk() {
  using namespace torch::indexing;
  if (!num_buffered_frames) {
    return {};
  }
  torch::Tensor chunk = chunks.front();
  double pts_val = double(pts.front()) * time_base.num / time_base.den;
  chunks.pop_front();
  pts.pop_front();
  if (num_buffered_frames < frames_per_chunk) {
    chunk = chunk.index({Slice(None, num_buffered_frames)});
  }
  num_buffered_frames -= chunk.size(0);
  return {Chunk{chunk, pts_val}};
}

template <typename Converter>
void ChunkedBuffer<Converter>::flush() {
  num_buffered_frames = 0;
  chunks.clear();
}

std::unique_ptr<Buffer> get_chunked_buffer(
    AVRational tb,
    int fpc,
    int num_chunks,
    AVSampleFormat fmt,
    int channels) {
  switch (fmt) {
    case AV_SAMPLE_FMT_U8: {
      using Converter = AudioConverter<torch::kUInt8, false>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    case AV_SAMPLE_FMT_S16: {
      using Converter = AudioConverter<torch::kInt16, false>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    case AV_SAMPLE_FMT_S32: {
      using Converter = AudioConverter<torch::kInt32, false>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    case AV_SAMPLE_FMT_S64: {
      using Converter = AudioConverter<torch::kInt64, false>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    case AV_SAMPLE_FMT_FLT: {
      using Converter = AudioConverter<torch::kFloat32, false>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    case AV_SAMPLE_FMT_DBL: {
      using Converter = AudioConverter<torch::kFloat64, false>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    case AV_SAMPLE_FMT_U8P: {
      using Converter = AudioConverter<torch::kUInt8, true>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    case AV_SAMPLE_FMT_S16P: {
      using Converter = AudioConverter<torch::kInt16, true>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    case AV_SAMPLE_FMT_S32P: {
      using Converter = AudioConverter<torch::kInt32, true>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    case AV_SAMPLE_FMT_S64P: {
      using Converter = AudioConverter<torch::kInt64, true>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    case AV_SAMPLE_FMT_FLTP: {
      using Converter = AudioConverter<torch::kFloat32, true>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    case AV_SAMPLE_FMT_DBLP: {
      using Converter = AudioConverter<torch::kFloat64, true>;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{channels});
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unexpected audio type:", av_get_sample_fmt_name(fmt));
  }
}

std::unique_ptr<Buffer> get_chunked_buffer(
    AVRational tb,
    int fpc,
    int num_chunks,
    AVPixelFormat fmt,
    int h,
    int w,
    const torch::Device& device) {
  if (device.type() == at::DeviceType::CUDA) {
#ifndef USE_CUDA
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        false,
        "USE_CUDA is not defined, and it should be guarded before here.");
#else
    switch (fmt) {
      case AV_PIX_FMT_NV12: {
        using Conv = NV12CudaConverter;
        return std::make_unique<ChunkedBuffer<Conv>>(
            tb, fpc, num_chunks, Conv{h, w, device});
      }
      case AV_PIX_FMT_P010: {
        using Conv = P010CudaConverter;
        return std::make_unique<ChunkedBuffer<Conv>>(
            tb, fpc, num_chunks, Conv{h, w, device});
      }
      case AV_PIX_FMT_P016: {
        TORCH_CHECK(
            false,
            "Unsupported video format found in CUDA HW: ",
            av_get_pix_fmt_name(fmt));
      }
      default: {
        TORCH_CHECK(
            false,
            "Unexpected video format found in CUDA HW: ",
            av_get_pix_fmt_name(fmt));
      }
    }
#endif
  }

  switch (fmt) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24: {
      using Converter = InterlacedImageConverter;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{h, w, 3});
    }
    case AV_PIX_FMT_ARGB:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_ABGR:
    case AV_PIX_FMT_BGRA: {
      using Converter = InterlacedImageConverter;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{h, w, 4});
    }
    case AV_PIX_FMT_GRAY8: {
      using Converter = InterlacedImageConverter;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{h, w, 1});
    }
    case AV_PIX_FMT_RGB48LE: {
      using Converter = Interlaced16BitImageConverter;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{h, w, 3});
    }
    case AV_PIX_FMT_YUV444P: {
      using Converter = PlanarImageConverter;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{h, w, 3});
    }
    case AV_PIX_FMT_YUV420P: {
      using Converter = YUV420PConverter;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{h, w});
    }
    case AV_PIX_FMT_NV12: {
      using Converter = NV12Converter;
      return std::make_unique<ChunkedBuffer<Converter>>(
          tb, fpc, num_chunks, Converter{h, w});
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Unexpected video format found: ", av_get_pix_fmt_name(fmt));
    }
  }
}

} // namespace torchaudio::io::detail
