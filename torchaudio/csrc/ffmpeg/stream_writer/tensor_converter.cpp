#include <torchaudio/csrc/ffmpeg/stream_writer/tensor_converter.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace torchaudio::io {

namespace {

using InitFunc = TensorConverter::InitFunc;
using ConvertFunc = TensorConverter::ConvertFunc;

////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////

void validate_audio_input(
    const torch::Tensor& t,
    AVFrame* buffer,
    c10::ScalarType dtype) {
  TORCH_CHECK(
      t.dtype().toScalarType() == dtype,
      "Expected ",
      dtype,
      " type. Found: ",
      t.dtype().toScalarType());
  TORCH_CHECK(t.device().is_cpu(), "Input tensor has to be on CPU.");
  TORCH_CHECK(t.dim() == 2, "Input Tensor has to be 2D.");
  TORCH_CHECK(
      t.size(1) == buffer->channels,
      "Expected waveform with ",
      buffer->channels,
      " channels. Found ",
      t.size(1));
}

// 2D (time, channel) and contiguous.
void convert_func_(const torch::Tensor& chunk, AVFrame* buffer) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(chunk.dim() == 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(chunk.size(1) == buffer->channels);

  // TODO: make writable
  // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00334
  TORCH_CHECK(av_frame_is_writable(buffer), "frame is not writable.");

  auto byte_size = chunk.numel() * chunk.element_size();
  memcpy(buffer->data[0], chunk.data_ptr(), byte_size);
  buffer->nb_samples = static_cast<int>(chunk.size(0));
}

std::pair<InitFunc, ConvertFunc> get_audio_func(AVFrame* buffer) {
  auto dtype = [&]() -> c10::ScalarType {
    switch (static_cast<AVSampleFormat>(buffer->format)) {
      case AV_SAMPLE_FMT_U8:
        return c10::ScalarType::Byte;
      case AV_SAMPLE_FMT_S16:
        return c10::ScalarType::Short;
      case AV_SAMPLE_FMT_S32:
        return c10::ScalarType::Int;
      case AV_SAMPLE_FMT_S64:
        return c10::ScalarType::Long;
      case AV_SAMPLE_FMT_FLT:
        return c10::ScalarType::Float;
      case AV_SAMPLE_FMT_DBL:
        return c10::ScalarType::Double;
      default:
        TORCH_INTERNAL_ASSERT(
            false, "Audio encoding process is not properly configured.");
    }
  }();

  InitFunc init_func = [=](const torch::Tensor& tensor, AVFrame* buffer) {
    validate_audio_input(tensor, buffer, dtype);
    return tensor.contiguous();
  };
  return {init_func, convert_func_};
}

////////////////////////////////////////////////////////////////////////////////
// Video
////////////////////////////////////////////////////////////////////////////////

void validate_video_input(
    const torch::Tensor& t,
    AVFrame* buffer,
    int num_channels) {
  if (buffer->hw_frames_ctx) {
    TORCH_CHECK(t.device().is_cuda(), "Input tensor has to be on CUDA.");
  } else {
    TORCH_CHECK(t.device().is_cpu(), "Input tensor has to be on CPU.");
  }
  TORCH_CHECK(
      t.dtype().toScalarType() == c10::ScalarType::Byte,
      "Expected Tensor of uint8 type.");

  TORCH_CHECK(t.dim() == 4, "Input Tensor has to be 4D.");
  TORCH_CHECK(
      t.size(1) == num_channels && t.size(2) == buffer->height &&
          t.size(3) == buffer->width,
      "Expected tensor with shape (N, ",
      num_channels,
      ", ",
      buffer->height,
      ", ",
      buffer->width,
      ") (NCHW format). Found ",
      t.sizes());
}

// NCHW ->NHWC, ensure contiguous
torch::Tensor init_interlaced(const torch::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.dim() == 4);
  return tensor.permute({0, 2, 3, 1}).contiguous();
}

// Keep NCHW, ensure contiguous
torch::Tensor init_planar(const torch::Tensor& tensor) {
  return tensor.contiguous();
}

// Interlaced video
// Each frame is composed of one plane, and color components for each pixel are
// collocated.
// The memory layout is 1D linear, interpretated as following.
//
//   |<----- linesize[0] ------>|
//   |<-- stride -->|
//      0   1 ...   W
// 0: RGB RGB ... RGB PAD ... PAD
// 1: RGB RGB ... RGB PAD ... PAD
//            ...
// H: RGB RGB ... RGB PAD ... PAD
void write_interlaced_video(
    const torch::Tensor& frame,
    AVFrame* buffer,
    int num_channels) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.dim() == 4);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(0) == 1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(1) == buffer->height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(2) == buffer->width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(3) == num_channels);

  // TODO: writable
  // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00472
  TORCH_INTERNAL_ASSERT(av_frame_is_writable(buffer), "frame is not writable.");

  size_t stride = buffer->width * num_channels;
  uint8_t* src = frame.data_ptr<uint8_t>();
  uint8_t* dst = buffer->data[0];
  for (int h = 0; h < buffer->height; ++h) {
    std::memcpy(dst, src, stride);
    src += stride;
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
  const auto num_colors =
      av_pix_fmt_desc_get((AVPixelFormat)buffer->format)->nb_components;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.dim() == 4);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(0) == 1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(1) == num_colors);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(2), buffer->height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(3), buffer->width);

  // TODO: writable
  // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00472
  TORCH_INTERNAL_ASSERT(av_frame_is_writable(buffer), "frame is not writable.");

  for (int j = 0; j < num_colors; ++j) {
    uint8_t* src = frame.index({0, j}).data_ptr<uint8_t>();
    uint8_t* dst = buffer->data[j];
    for (int h = 0; h < buffer->height; ++h) {
      memcpy(dst, src, buffer->width);
      src += buffer->width;
      dst += buffer->linesize[j];
    }
  }
}

void write_interlaced_video_cuda(
    const torch::Tensor& frame,
    AVFrame* buffer,
    int num_channels) {
#ifndef USE_CUDA
  TORCH_CHECK(
      false,
      "torchaudio is not compiled with CUDA support. Hardware acceleration is not available.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.dim() == 4);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(0) == 1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(1) == buffer->height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(2) == buffer->width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(3) == num_channels);
  size_t spitch = buffer->width * num_channels;
  if (cudaSuccess !=
      cudaMemcpy2D(
          (void*)(buffer->data[0]),
          buffer->linesize[0],
          (const void*)(frame.data_ptr<uint8_t>()),
          spitch,
          spitch,
          buffer->height,
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
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.dim() == 4);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(0) == 1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(1) == num_planes);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(2) == buffer->height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(3) == buffer->width);
  for (int j = 0; j < num_planes; ++j) {
    if (cudaSuccess !=
        cudaMemcpy2D(
            (void*)(buffer->data[j]),
            buffer->linesize[j],
            (const void*)(frame.index({0, j}).data_ptr<uint8_t>()),
            buffer->width,
            buffer->width,
            buffer->height,
            cudaMemcpyDeviceToDevice)) {
      TORCH_CHECK(false, "Failed to copy pixel data from CUDA tensor.");
    }
  }
#endif
}

std::pair<InitFunc, ConvertFunc> get_video_func(AVFrame* buffer) {
  if (buffer->hw_frames_ctx) {
    auto frames_ctx = (AVHWFramesContext*)(buffer->hw_frames_ctx->data);
    auto sw_pix_fmt = frames_ctx->sw_format;
    switch (sw_pix_fmt) {
      // Note:
      // RGB0 / BGR0 expects 4 channel, but neither
      // av_pix_fmt_desc_get(pix_fmt)->nb_components
      // or av_pix_fmt_count_planes(pix_fmt) returns 4.
      case AV_PIX_FMT_RGB0:
      case AV_PIX_FMT_BGR0: {
        ConvertFunc convert_func = [](const torch::Tensor& t, AVFrame* f) {
          write_interlaced_video_cuda(t, f, 4);
        };
        InitFunc init_func = [](const torch::Tensor& t, AVFrame* f) {
          validate_video_input(t, f, 4);
          return init_interlaced(t);
        };
        return {init_func, convert_func};
      }
      case AV_PIX_FMT_GBRP:
      case AV_PIX_FMT_GBRP16LE:
      case AV_PIX_FMT_YUV444P:
      case AV_PIX_FMT_YUV444P16LE: {
        ConvertFunc convert_func = [](const torch::Tensor& t, AVFrame* f) {
          write_planar_video_cuda(t, f, 3);
        };
        InitFunc init_func = [](const torch::Tensor& t, AVFrame* f) {
          validate_video_input(t, f, 3);
          return init_planar(t);
        };
        return {init_func, convert_func};
      }
      default:
        TORCH_CHECK(
            false,
            "Unexpected pixel format for CUDA: ",
            av_get_pix_fmt_name(sw_pix_fmt));
    }
  }

  auto pix_fmt = static_cast<AVPixelFormat>(buffer->format);
  switch (pix_fmt) {
    case AV_PIX_FMT_GRAY8:
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24: {
      int channels = av_pix_fmt_desc_get(pix_fmt)->nb_components;
      InitFunc init_func = [=](const torch::Tensor& t, AVFrame* f) {
        validate_video_input(t, f, channels);
        return init_interlaced(t);
      };
      ConvertFunc convert_func = [=](const torch::Tensor& t, AVFrame* f) {
        write_interlaced_video(t, f, channels);
      };
      return {init_func, convert_func};
    }
    case AV_PIX_FMT_YUV444P: {
      InitFunc init_func = [](const torch::Tensor& t, AVFrame* f) {
        validate_video_input(t, f, 3);
        return init_planar(t);
      };
      ConvertFunc convert_func = [](const torch::Tensor& t, AVFrame* f) {
        write_planar_video(t, f, 3);
      };
      return {init_func, convert_func};
    }
    default:
      TORCH_CHECK(
          false, "Unexpected pixel format: ", av_get_pix_fmt_name(pix_fmt));
  }
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// TensorConverter
////////////////////////////////////////////////////////////////////////////////

TensorConverter::TensorConverter(AVMediaType type, AVFrame* buf, int buf_size)
    : buffer(buf), buffer_size(buf_size) {
  switch (type) {
    case AVMEDIA_TYPE_AUDIO:
      std::tie(init_func, convert_func) = get_audio_func(buffer);
      break;
    case AVMEDIA_TYPE_VIDEO:
      std::tie(init_func, convert_func) = get_video_func(buffer);
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unsupported media type: ", av_get_media_type_string(type));
  }
}

using Generator = TensorConverter::Generator;

Generator TensorConverter::convert(const torch::Tensor& t) {
  return Generator{init_func(t, buffer), buffer, convert_func, buffer_size};
}

////////////////////////////////////////////////////////////////////////////////
// Generator
////////////////////////////////////////////////////////////////////////////////

using Iterator = Generator::Iterator;

Generator::Generator(
    torch::Tensor frames_,
    AVFrame* buff,
    ConvertFunc& func,
    int64_t step_)
    : frames(std::move(frames_)),
      buffer(buff),
      convert_func(func),
      step(step_) {}

Iterator Generator::begin() const {
  return Iterator{frames, buffer, convert_func, step};
}

int64_t Generator::end() const {
  return frames.size(0);
}

////////////////////////////////////////////////////////////////////////////////
// Iterator
////////////////////////////////////////////////////////////////////////////////

Iterator::Iterator(
    const torch::Tensor frames_,
    AVFrame* buffer_,
    ConvertFunc& convert_func_,
    int64_t step_)
    : frames(frames_),
      buffer(buffer_),
      convert_func(convert_func_),
      step(step_) {}

Iterator& Iterator::operator++() {
  i += step;
  return *this;
}

AVFrame* Iterator::operator*() const {
  using namespace torch::indexing;
  convert_func(frames.index({Slice{i, i + step}}), buffer);
  return buffer;
}

bool Iterator::operator!=(const int64_t end) const {
  // This is used for detecting the end of iteraton.
  // For audio, iteration is done by
  return i < end;
}

} // namespace torchaudio::io
