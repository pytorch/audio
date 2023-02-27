#include <torchaudio/csrc/ffmpeg/stream_writer/output_stream.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace torchaudio::io {

OutputStream::OutputStream(
    AVFormatContext* format_ctx,
    AVCodecContextPtr&& codec_ctx_,
    std::unique_ptr<FilterGraph>&& filter_,
    AVFramePtr&& src_frame_)
    : codec_ctx(std::move(codec_ctx_)),
      encoder(format_ctx, codec_ctx),
      filter(std::move(filter_)),
      src_frame(std::move(src_frame_)),
      dst_frame(),
      num_frames(0) {}

AudioOutputStream::AudioOutputStream(
    AVFormatContext* format_ctx,
    AVCodecContextPtr&& codec_ctx,
    std::unique_ptr<FilterGraph>&& filter,
    AVFramePtr&& src_frame,
    int64_t frame_capacity_)
    : OutputStream(
          format_ctx,
          std::move(codec_ctx),
          std::move(filter),
          std::move(src_frame)),
      frame_capacity(frame_capacity_) {}

VideoOutputStream::VideoOutputStream(
    AVFormatContext* format_ctx,
    AVCodecContextPtr&& codec_ctx,
    std::unique_ptr<FilterGraph>&& filter,
    AVFramePtr&& src_frame,
    AVBufferRefPtr&& hw_device_ctx_,
    AVBufferRefPtr&& hw_frame_ctx_)
    : OutputStream(
          format_ctx,
          std::move(codec_ctx),
          std::move(filter),
          std::move(src_frame)),
      hw_device_ctx(std::move(hw_device_ctx_)),
      hw_frame_ctx(std::move(hw_frame_ctx_)) {}

void OutputStream::process_frame(AVFrame* src) {
  if (!filter) {
    encoder.encode(src);
    return;
  }
  int ret = filter->add_frame(src);
  while (ret >= 0) {
    ret = filter->get_frame(dst_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      if (ret == AVERROR_EOF) {
        encoder.encode(nullptr);
      }
      break;
    }
    if (ret >= 0) {
      encoder.encode(dst_frame);
    }
    av_frame_unref(dst_frame);
  }
}

void OutputStream::flush() {
  process_frame(nullptr);
}

namespace {

void validate_audio_input(
    enum AVSampleFormat fmt,
    AVCodecContext* ctx,
    const torch::Tensor& t) {
  auto dtype = t.dtype().toScalarType();
  switch (fmt) {
    case AV_SAMPLE_FMT_U8:
      TORCH_CHECK(
          dtype == c10::ScalarType::Byte, "Expected Tensor of uint8 type.");
      break;
    case AV_SAMPLE_FMT_S16:
      TORCH_CHECK(
          dtype == c10::ScalarType::Short, "Expected Tensor of int16 type.");
      break;
    case AV_SAMPLE_FMT_S32:
      TORCH_CHECK(
          dtype == c10::ScalarType::Int, "Expected Tensor of int32 type.");
      break;
    case AV_SAMPLE_FMT_S64:
      TORCH_CHECK(
          dtype == c10::ScalarType::Long, "Expected Tensor of int64 type.");
      break;
    case AV_SAMPLE_FMT_FLT:
      TORCH_CHECK(
          dtype == c10::ScalarType::Float, "Expected Tensor of float32 type.");
      break;
    case AV_SAMPLE_FMT_DBL:
      TORCH_CHECK(
          dtype == c10::ScalarType::Double, "Expected Tensor of float64 type.");
      break;
    default:
      TORCH_CHECK(
          false,
          "Internal error: Audio encoding stream is not properly configured.");
  }
  TORCH_CHECK(t.device().is_cpu(), "Input tensor has to be on CPU.");
  TORCH_CHECK(t.dim() == 2, "Input Tensor has to be 2D.");
  const auto num_channels = t.size(1);
  TORCH_CHECK(
      num_channels == ctx->channels,
      "Expected waveform with ",
      ctx->channels,
      " channels. Found ",
      num_channels);
}

} // namespace

void AudioOutputStream::write_chunk(const torch::Tensor& waveform) {
  validate_audio_input(
      static_cast<AVSampleFormat>(src_frame->format), codec_ctx, waveform);

  AVRational time_base{1, codec_ctx->sample_rate};

  using namespace torch::indexing;
  AT_DISPATCH_ALL_TYPES(waveform.scalar_type(), "write_audio_frames", [&] {
    for (int64_t i = 0; i < waveform.size(0); i += frame_capacity) {
      auto chunk = waveform.index({Slice(i, i + frame_capacity), Slice()});
      auto num_valid_frames = chunk.size(0);
      auto byte_size = chunk.numel() * chunk.element_size();
      chunk = chunk.reshape({-1}).contiguous();

      // TODO: make writable
      // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00334
      TORCH_CHECK(
          av_frame_is_writable(src_frame),
          "Internal Error: frame is not writable.");

      memcpy(
          src_frame->data[0],
          static_cast<void*>(chunk.data_ptr<scalar_t>()),
          byte_size);
      src_frame->pts =
          av_rescale_q(num_frames, time_base, codec_ctx->time_base);
      src_frame->nb_samples = num_valid_frames;
      num_frames += num_valid_frames;

      process_frame(src_frame);
    }
  });
}

namespace {

void validate_video_input(
    enum AVPixelFormat fmt,
    AVCodecContext* ctx,
    const torch::Tensor& t) {
  auto dtype = t.dtype().toScalarType();
  TORCH_CHECK(dtype == c10::ScalarType::Byte, "Expected Tensor of uint8 type.");
  TORCH_CHECK(t.dim() == 4, "Input Tensor has to be 4D.");

  // Note: the number of color components is not same as the number of planes.
  // For example, YUV420P has only two planes. U and V are in the second plane.
  int num_color_components = av_pix_fmt_desc_get(fmt)->nb_components;

  const auto channels = t.size(1);
  const auto height = t.size(2);
  const auto width = t.size(3);
  TORCH_CHECK(
      channels == num_color_components && height == ctx->height &&
          width == ctx->width,
      "Expected tensor with shape (N, ",
      num_color_components,
      ", ",
      ctx->height,
      ", ",
      ctx->width,
      ") (NCHW format). Found ",
      t.sizes());
}

#ifdef USE_CUDA
void write_interlaced_video_cuda(
    OutputStream& os,
    const torch::Tensor& frames,
    bool pad_extra) {
  const auto num_frames = frames.size(0);
  const auto num_channels = frames.size(1);
  const auto height = frames.size(2);
  const auto width = frames.size(3);
  const auto num_channels_buffer = num_channels + (pad_extra ? 1 : 0);

  using namespace torch::indexing;
  torch::Tensor buffer =
      torch::empty({height, width, num_channels_buffer}, frames.options());
  size_t spitch = width * num_channels_buffer;
  for (int i = 0; i < num_frames; ++i) {
    // Slice frame as HWC
    auto chunk = frames.index({i}).permute({1, 2, 0});
    buffer.index_put_({"...", Slice(0, num_channels)}, chunk);

    if (cudaSuccess !=
        cudaMemcpy2D(
            (void*)(os.src_frame->data[0]),
            os.src_frame->linesize[0],
            (const void*)(buffer.data_ptr<uint8_t>()),
            spitch,
            spitch,
            height,
            cudaMemcpyDeviceToDevice)) {
      TORCH_CHECK(false, "Failed to copy pixel data from CUDA tensor.");
    }
    os.src_frame->pts = os.num_frames;
    os.num_frames += 1;
    os.process_frame(os.src_frame);
  }
}

void write_planar_video_cuda(
    OutputStream& os,
    const torch::Tensor& frames,
    int num_planes) {
  const auto num_frames = frames.size(0);
  const auto height = frames.size(2);
  const auto width = frames.size(3);

  using namespace torch::indexing;
  torch::Tensor buffer = torch::empty({height, width}, frames.options());
  for (int i = 0; i < num_frames; ++i) {
    for (int j = 0; j < num_planes; ++j) {
      buffer.index_put_({"..."}, frames.index({i, j}));
      if (cudaSuccess !=
          cudaMemcpy2D(
              (void*)(os.src_frame->data[j]),
              os.src_frame->linesize[j],
              (const void*)(buffer.data_ptr<uint8_t>()),
              width,
              width,
              height,
              cudaMemcpyDeviceToDevice)) {
        TORCH_CHECK(false, "Failed to copy pixel data from CUDA tensor.");
      }
    }
    os.src_frame->pts = os.num_frames;
    os.num_frames += 1;
    os.process_frame(os.src_frame);
  }
}
#endif

// Interlaced video
// Each frame is composed of one plane, and color components for each pixel are
// collocated.
// The memory layout is 1D linear, interpretated as following.
//
//    |<----- linesize[0] ----->|
//      0   1 ...   W
// 0: RGB RGB ... RGB PAD ... PAD
// 1: RGB RGB ... RGB PAD ... PAD
//            ...
// H: RGB RGB ... RGB PAD ... PAD
void write_interlaced_video(OutputStream& os, const torch::Tensor& frames) {
  const auto num_frames = frames.size(0);
  const auto num_channels = frames.size(1);
  const auto height = frames.size(2);
  const auto width = frames.size(3);

  using namespace torch::indexing;
  size_t stride = width * num_channels;
  for (int i = 0; i < num_frames; ++i) {
    // TODO: writable
    // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00472
    TORCH_CHECK(
        av_frame_is_writable(os.src_frame),
        "Internal Error: frame is not writable.");

    // CHW -> HWC
    auto chunk =
        frames.index({i}).permute({1, 2, 0}).reshape({-1}).contiguous();

    uint8_t* src = chunk.data_ptr<uint8_t>();
    uint8_t* dst = os.src_frame->data[0];
    for (int h = 0; h < height; ++h) {
      std::memcpy(dst, src, stride);
      src += width * num_channels;
      dst += os.src_frame->linesize[0];
    }
    os.src_frame->pts = os.num_frames;
    os.num_frames += 1;

    os.process_frame(os.src_frame);
  }
}

// Planar video
// Each frame is composed of multiple planes.
// One plane can contain one of more color components.
// (but at the moment only accept formats without subsampled color components)
//
// The memory layout is interpreted as follow
//
//    |<----- linesize[0] ----->|
//       0   1 ...  W1
//  0:   Y   Y ...   Y PAD ... PAD
//  1:   Y   Y ...   Y PAD ... PAD
//             ...
// H1:   Y   Y ...   Y PAD ... PAD
//
//    |<--- linesize[1] ---->|
//       0 ...  W2
//  0:  UV ...  UV PAD ... PAD
//  1:  UV ...  UV PAD ... PAD
//         ...
// H2:  UV ...  UV PAD ... PAD
//
void write_planar_video(
    OutputStream& os,
    const torch::Tensor& frames,
    int num_planes) {
  const auto num_frames = frames.size(0);
  const auto height = frames.size(2);
  const auto width = frames.size(3);

  using namespace torch::indexing;
  for (int i = 0; i < num_frames; ++i) {
    // TODO: writable
    // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00472
    TORCH_CHECK(
        av_frame_is_writable(os.src_frame),
        "Internal Error: frame is not writable.");

    for (int j = 0; j < num_planes; ++j) {
      auto chunk = frames.index({i, j}).contiguous();

      uint8_t* src = chunk.data_ptr<uint8_t>();
      uint8_t* dst = os.src_frame->data[j];
      for (int h = 0; h < height; ++h) {
        memcpy(dst, src, width);
        src += width;
        dst += os.src_frame->linesize[j];
      }
    }
    os.src_frame->pts = os.num_frames;
    os.num_frames += 1;

    os.process_frame(os.src_frame);
  }
}

} // namespace

void VideoOutputStream::write_chunk(const torch::Tensor& frames) {
  enum AVPixelFormat fmt = static_cast<AVPixelFormat>(src_frame->format);

#ifdef USE_CUDA
  if (fmt == AV_PIX_FMT_CUDA) {
    TORCH_CHECK(frames.device().is_cuda(), "Input tensor has to be on CUDA.");
    enum AVPixelFormat sw_fmt = codec_ctx->sw_pix_fmt;
    validate_video_input(sw_fmt, codec_ctx, frames);
    switch (sw_fmt) {
      case AV_PIX_FMT_RGB0:
      case AV_PIX_FMT_BGR0:
        write_interlaced_video_cuda(*this, frames, true);
        return;
      case AV_PIX_FMT_GBRP:
      case AV_PIX_FMT_GBRP16LE:
      case AV_PIX_FMT_YUV444P:
      case AV_PIX_FMT_YUV444P16LE:
        write_planar_video_cuda(*this, frames, av_pix_fmt_count_planes(sw_fmt));
        return;
      default:
        TORCH_CHECK(
            false,
            "Unexpected pixel format for CUDA: ",
            av_get_pix_fmt_name(sw_fmt));
    }
  }
#endif

  TORCH_CHECK(frames.device().is_cpu(), "Input tensor has to be on CPU.");
  validate_video_input(fmt, codec_ctx, frames);
  switch (fmt) {
    case AV_PIX_FMT_GRAY8:
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
      write_interlaced_video(*this, frames);
      return;
    case AV_PIX_FMT_YUV444P:
      write_planar_video(*this, frames, av_pix_fmt_count_planes(fmt));
      return;
    default:
      TORCH_CHECK(false, "Unexpected pixel format: ", av_get_pix_fmt_name(fmt));
  }
}

} // namespace torchaudio::io
