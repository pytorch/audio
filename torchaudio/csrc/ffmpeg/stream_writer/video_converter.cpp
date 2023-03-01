#include <torchaudio/csrc/ffmpeg/stream_writer/video_converter.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace torchaudio::io {

////////////////////////////////////////////////////////////////////////////////
// VideoTensorConverter
////////////////////////////////////////////////////////////////////////////////

using InitFunc = VideoTensorConverter::InitFunc;
using ConvertFunc = Generator::ConvertFunc;

namespace {

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
void write_interlaced_video(const torch::Tensor& frame, AVFrame* buffer) {
  const auto height = frame.size(0);
  const auto width = frame.size(1);
  const auto num_channels = frame.size(2);

  size_t stride = width * num_channels;
  // TODO: writable
  // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00472
  TORCH_INTERNAL_ASSERT(av_frame_is_writable(buffer), "frame is not writable.");

  uint8_t* src = frame.data_ptr<uint8_t>();
  uint8_t* dst = buffer->data[0];
  for (int h = 0; h < height; ++h) {
    std::memcpy(dst, src, stride);
    src += width * num_channels;
    dst += buffer->linesize[0];
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
    const torch::Tensor& frame,
    AVFrame* buffer,
    int num_planes) {
  const auto height = frame.size(1);
  const auto width = frame.size(2);

  // TODO: writable
  // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00472
  TORCH_INTERNAL_ASSERT(av_frame_is_writable(buffer), "frame is not writable.");

  for (int j = 0; j < num_planes; ++j) {
    uint8_t* src = frame.index({j}).data_ptr<uint8_t>();
    uint8_t* dst = buffer->data[j];
    for (int h = 0; h < height; ++h) {
      memcpy(dst, src, width);
      src += width;
      dst += buffer->linesize[j];
    }
  }
}

void write_interlaced_video_cuda(
    const torch::Tensor& frame,
    AVFrame* buffer,
    bool pad_extra) {
#ifndef USE_CUDA
  TORCH_CHECK(
      false,
      "torchaudio is not compiled with CUDA support. Hardware acceleration is not available.");
#else
  const auto height = frame.size(0);
  const auto width = frame.size(1);
  const auto num_channels = frame.size(2) + (pad_extra ? 1 : 0);
  size_t spitch = width * num_channels;
  if (cudaSuccess !=
      cudaMemcpy2D(
          (void*)(buffer->data[0]),
          buffer->linesize[0],
          (const void*)(frame.data_ptr<uint8_t>()),
          spitch,
          spitch,
          height,
          cudaMemcpyDeviceToDevice)) {
    TORCH_CHECK(false, "Failed to copy pixel data from CUDA tensor.");
  }
#endif
}

void write_planar_video_cuda(
    const torch::Tensor& frame,
    AVFrame* buffer,
    int num_planes) {
#ifndef USE_CUDA
  TORCH_CHECK(
      false,
      "torchaudio is not compiled with CUDA support. Hardware acceleration is not available.");
#else
  const auto height = frame.size(1);
  const auto width = frame.size(2);
  for (int j = 0; j < num_planes; ++j) {
    if (cudaSuccess !=
        cudaMemcpy2D(
            (void*)(buffer->data[j]),
            buffer->linesize[j],
            (const void*)(frame.index({j}).data_ptr<uint8_t>()),
            width,
            width,
            height,
            cudaMemcpyDeviceToDevice)) {
      TORCH_CHECK(false, "Failed to copy pixel data from CUDA tensor.");
    }
  }
#endif
}

// NCHW ->NHWC, ensure contiguous
torch::Tensor init_interlaced(const torch::Tensor& tensor) {
  return tensor.permute({0, 2, 3, 1}).contiguous();
}

// Keep NCHW, ensure contiguous
torch::Tensor init_planar(const torch::Tensor& tensor) {
  return tensor.contiguous();
}

std::pair<InitFunc, ConvertFunc> get_func(
    enum AVPixelFormat pix_fmt,
    enum AVPixelFormat sw_pix_fmt) {
  using namespace std::placeholders;
  if (pix_fmt == AV_PIX_FMT_CUDA) {
    switch (sw_pix_fmt) {
      case AV_PIX_FMT_RGB0:
      case AV_PIX_FMT_BGR0: {
        ConvertFunc convert_func = [](const torch::Tensor& t, AVFrame* f) {
          write_interlaced_video_cuda(t, f, true);
        };
        return {init_interlaced, convert_func};
      }
      case AV_PIX_FMT_GBRP:
      case AV_PIX_FMT_GBRP16LE:
      case AV_PIX_FMT_YUV444P:
      case AV_PIX_FMT_YUV444P16LE: {
        auto num_planes = av_pix_fmt_count_planes(sw_pix_fmt);
        ConvertFunc convert_func = [=](const torch::Tensor& t, AVFrame* f) {
          write_planar_video_cuda(t, f, num_planes);
        };
        return {init_planar, convert_func};
      }
      default:
        TORCH_CHECK(
            false,
            "Unexpected pixel format for CUDA: ",
            av_get_pix_fmt_name(sw_pix_fmt));
    }
  }

  switch (pix_fmt) {
    case AV_PIX_FMT_GRAY8:
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24: {
      return {init_interlaced, write_interlaced_video};
    }
    case AV_PIX_FMT_YUV444P: {
      auto num_planes = av_pix_fmt_count_planes(pix_fmt);
      ConvertFunc convert_func = [=](const torch::Tensor& t, AVFrame* f) {
        write_planar_video(t, f, num_planes);
      };
      return {init_planar, convert_func};
    }
    default:
      TORCH_CHECK(
          false, "Unexpected pixel format: ", av_get_pix_fmt_name(pix_fmt));
  }
}

AVFramePtr get_video_frame(AVPixelFormat src_fmt, AVCodecContext* codec_ctx) {
  AVFramePtr frame{};
  if (codec_ctx->pix_fmt == AV_PIX_FMT_CUDA) {
    int ret = av_hwframe_get_buffer(codec_ctx->hw_frames_ctx, frame, 0);
    TORCH_CHECK(ret >= 0, "Failed to fetch CUDA frame: ", av_err2string(ret));
  } else {
    frame->format = src_fmt;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;

    int ret = av_frame_get_buffer(frame, 0);
    TORCH_CHECK(
        ret >= 0,
        "Error allocating a video buffer (",
        av_err2string(ret),
        ").");
  }
  frame->pts = 0;
  return frame;
}

void validate_video_input(
    enum AVPixelFormat fmt,
    AVCodecContext* ctx,
    const torch::Tensor& t) {
  if (fmt == AV_PIX_FMT_CUDA) {
    TORCH_CHECK(t.device().is_cuda(), "Input tensor has to be on CUDA.");
    fmt = ctx->sw_pix_fmt;
  } else {
    TORCH_CHECK(t.device().is_cpu(), "Input tensor has to be on CPU.");
  }

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

} // namespace

VideoTensorConverter::VideoTensorConverter(
    enum AVPixelFormat src_fmt_,
    AVCodecContext* codec_ctx_)
    : src_fmt(src_fmt_),
      codec_ctx(codec_ctx_),
      buffer(get_video_frame(src_fmt_, codec_ctx_)) {
  std::tie(init_func, convert_func) = get_func(src_fmt, codec_ctx->sw_pix_fmt);
}

Generator VideoTensorConverter::convert(const torch::Tensor& frames) {
  validate_video_input(src_fmt, codec_ctx, frames);
  return Generator{init_func(frames), buffer, convert_func};
}

} // namespace torchaudio::io
