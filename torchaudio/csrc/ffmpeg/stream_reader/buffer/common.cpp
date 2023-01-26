#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/common.h>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace torchaudio {
namespace io {
namespace detail {

torch::Tensor convert_audio(AVFrame* pFrame) {
  // ref: https://ffmpeg.org/doxygen/4.1/filter__audio_8c_source.html#l00215
  AVSampleFormat format = static_cast<AVSampleFormat>(pFrame->format);
  int num_channels = pFrame->channels;
  int bps = av_get_bytes_per_sample(format);

  // Note
  // FFMpeg's `nb_samples` represnts the number of samples par channel.
  // This corresponds to `num_frames` in torchaudio's notation.
  // Also torchaudio uses `num_samples` as the number of samples
  // across channels.
  int num_frames = pFrame->nb_samples;

  int is_planar = av_sample_fmt_is_planar(format);
  int num_planes = is_planar ? num_channels : 1;
  int plane_size = bps * num_frames * (is_planar ? 1 : num_channels);
  std::vector<int64_t> shape = is_planar
      ? std::vector<int64_t>{num_channels, num_frames}
      : std::vector<int64_t>{num_frames, num_channels};

  torch::Tensor t;
  uint8_t* ptr = nullptr;
  switch (format) {
    case AV_SAMPLE_FMT_U8:
    case AV_SAMPLE_FMT_U8P: {
      t = torch::empty(shape, torch::kUInt8);
      ptr = t.data_ptr<uint8_t>();
      break;
    }
    case AV_SAMPLE_FMT_S16:
    case AV_SAMPLE_FMT_S16P: {
      t = torch::empty(shape, torch::kInt16);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<int16_t>());
      break;
    }
    case AV_SAMPLE_FMT_S32:
    case AV_SAMPLE_FMT_S32P: {
      t = torch::empty(shape, torch::kInt32);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<int32_t>());
      break;
    }
    case AV_SAMPLE_FMT_S64:
    case AV_SAMPLE_FMT_S64P: {
      t = torch::empty(shape, torch::kInt64);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<int64_t>());
      break;
    }
    case AV_SAMPLE_FMT_FLT:
    case AV_SAMPLE_FMT_FLTP: {
      t = torch::empty(shape, torch::kFloat32);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<float>());
      break;
    }
    case AV_SAMPLE_FMT_DBL:
    case AV_SAMPLE_FMT_DBLP: {
      t = torch::empty(shape, torch::kFloat64);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<double>());
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported audio format: " +
              std::string(av_get_sample_fmt_name(format)));
  }
  for (int i = 0; i < num_planes; ++i) {
    memcpy(ptr, pFrame->extended_data[i], plane_size);
    ptr += plane_size;
  }
  if (is_planar) {
    t = t.t();
  }
  return t;
}

namespace {
torch::Tensor get_buffer(
    at::IntArrayRef shape,
    const torch::Device& device = torch::Device(torch::kCPU)) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .layout(torch::kStrided)
                     .device(device.type(), device.index());
  return torch::empty(shape, options);
}

std::tuple<torch::Tensor, bool> get_image_buffer(
    AVFrame* frame,
    int num_frames,
    const torch::Device& device) {
  auto fmt = static_cast<AVPixelFormat>(frame->format);
  const AVPixFmtDescriptor* desc = [&]() {
    if (fmt == AV_PIX_FMT_CUDA) {
      AVHWFramesContext* hwctx = (AVHWFramesContext*)frame->hw_frames_ctx->data;
      return av_pix_fmt_desc_get(hwctx->sw_format);
    }
    return av_pix_fmt_desc_get(fmt);
  }();
  int channels = desc->nb_components;

  // Note
  // AVPixFmtDescriptor::nb_components represents the number of
  // color components. This is different from the number of planes.
  //
  // For example, YUV420P has three color components Y, U and V, but
  // U and V are squashed into the same plane, so there are only
  // two planes.
  //
  // In our application, we cannot express the bare YUV420P as a
  // single tensor, so we convert it to 3 channel tensor.
  // For this reason, we use nb_components for the number of channels,
  // instead of the number of planes.
  //
  // The actual number of planes can be retrieved with
  // av_pix_fmt_count_planes.

  int height = frame->height;
  int width = frame->width;
  if (desc->flags & AV_PIX_FMT_FLAG_PLANAR) {
    auto buffer = get_buffer({num_frames, channels, height, width}, device);
    return std::make_tuple(buffer, true);
  }
  auto buffer = get_buffer({num_frames, height, width, channels}, device);
  return std::make_tuple(buffer, false);
}

void write_interlaced_image(AVFrame* pFrame, torch::Tensor& frame) {
  auto ptr = frame.data_ptr<uint8_t>();
  uint8_t* buf = pFrame->data[0];
  size_t height = frame.size(1);
  size_t stride = frame.size(2) * frame.size(3);
  for (int i = 0; i < height; ++i) {
    memcpy(ptr, buf, stride);
    buf += pFrame->linesize[0];
    ptr += stride;
  }
}

void write_planar_image(AVFrame* pFrame, torch::Tensor& frame) {
  int num_planes = static_cast<int>(frame.size(1));
  int height = static_cast<int>(frame.size(2));
  int width = static_cast<int>(frame.size(3));
  for (int i = 0; i < num_planes; ++i) {
    torch::Tensor plane = frame.index({0, i});
    uint8_t* tgt = plane.data_ptr<uint8_t>();
    uint8_t* src = pFrame->data[i];
    int linesize = pFrame->linesize[i];
    for (int h = 0; h < height; ++h) {
      memcpy(tgt, src, width);
      tgt += width;
      src += linesize;
    }
  }
}

void write_yuv420p(AVFrame* pFrame, torch::Tensor& yuv) {
  int height = static_cast<int>(yuv.size(2));
  int width = static_cast<int>(yuv.size(3));

  // Write Y plane directly
  {
    uint8_t* tgt = yuv.data_ptr<uint8_t>();
    uint8_t* src = pFrame->data[0];
    int linesize = pFrame->linesize[0];
    for (int h = 0; h < height; ++h) {
      memcpy(tgt, src, width);
      tgt += width;
      src += linesize;
    }
  }

  // Prepare intermediate UV plane
  torch::Tensor uv = get_buffer({1, 2, height / 2, width / 2});
  {
    uint8_t* tgt = uv.data_ptr<uint8_t>();
    uint8_t* src = pFrame->data[1];
    int linesize = pFrame->linesize[1];
    for (int h = 0; h < height / 2; ++h) {
      memcpy(tgt, src, width / 2);
      tgt += width / 2;
      src += linesize;
    }
    src = pFrame->data[2];
    linesize = pFrame->linesize[2];
    for (int h = 0; h < height / 2; ++h) {
      memcpy(tgt, src, width / 2);
      tgt += width / 2;
      src += linesize;
    }
  }
  // Upsample width and height
  namespace F = torch::nn::functional;
  using namespace torch::indexing;
  uv = F::interpolate(
      uv,
      F::InterpolateFuncOptions()
          .mode(torch::kNearest)
          .size(std::vector<int64_t>({height, width})));
  // Write to the UV plane
  // yuv[:, 1:] = uv
  yuv.index_put_({Slice(), Slice(1)}, uv);
}

void write_nv12_cpu(AVFrame* pFrame, torch::Tensor& yuv) {
  int height = static_cast<int>(yuv.size(2));
  int width = static_cast<int>(yuv.size(3));

  // Write Y plane directly
  {
    uint8_t* tgt = yuv.data_ptr<uint8_t>();
    uint8_t* src = pFrame->data[0];
    int linesize = pFrame->linesize[0];
    for (int h = 0; h < height; ++h) {
      memcpy(tgt, src, width);
      tgt += width;
      src += linesize;
    }
  }

  // Prepare intermediate UV plane
  torch::Tensor uv = get_buffer({1, height / 2, width / 2, 2});
  {
    uint8_t* tgt = uv.data_ptr<uint8_t>();
    uint8_t* src = pFrame->data[1];
    int linesize = pFrame->linesize[1];
    for (int h = 0; h < height / 2; ++h) {
      memcpy(tgt, src, width);
      tgt += width;
      src += linesize;
    }
  }

  // Upsample width and height
  namespace F = torch::nn::functional;
  using namespace torch::indexing;
  uv = F::interpolate(
      uv.permute({0, 3, 1, 2}),
      F::InterpolateFuncOptions()
          .mode(torch::kNearest)
          .size(std::vector<int64_t>({height, width})));
  // Write to the UV plane
  // yuv[:, 1:] = uv
  yuv.index_put_({Slice(), Slice(1)}, uv);
}

#ifdef USE_CUDA
void write_nv12_cuda(AVFrame* pFrame, torch::Tensor& yuv) {
  int height = static_cast<int>(yuv.size(2));
  int width = static_cast<int>(yuv.size(3));

  // Write Y plane directly
  {
    uint8_t* tgt = yuv.data_ptr<uint8_t>();
    CUdeviceptr src = (CUdeviceptr)pFrame->data[0];
    int linesize = pFrame->linesize[0];
    TORCH_CHECK(
        cudaSuccess ==
            cudaMemcpy2D(
                (void*)tgt,
                width,
                (const void*)src,
                linesize,
                width,
                height,
                cudaMemcpyDeviceToDevice),
        "Failed to copy Y plane to Cuda tensor.");
  }
  // Preapare intermediate UV planes
  torch::Tensor uv = get_buffer({1, height / 2, width / 2, 2}, yuv.device());
  {
    uint8_t* tgt = uv.data_ptr<uint8_t>();
    CUdeviceptr src = (CUdeviceptr)pFrame->data[1];
    int linesize = pFrame->linesize[1];
    TORCH_CHECK(
        cudaSuccess ==
            cudaMemcpy2D(
                (void*)tgt,
                width,
                (const void*)src,
                linesize,
                width,
                height / 2,
                cudaMemcpyDeviceToDevice),
        "Failed to copy UV plane to Cuda tensor.");
  }
  // Upsample width and height
  namespace F = torch::nn::functional;
  using namespace torch::indexing;
  uv = F::interpolate(
      uv.permute({0, 3, 1, 2}),
      F::InterpolateFuncOptions()
          .mode(torch::kNearest)
          .size(std::vector<int64_t>({height, width})));
  // Write to the UV plane
  // yuv[:, 1:] = uv
  yuv.index_put_({Slice(), Slice(1)}, uv);
}
#endif

void write_image(AVFrame* frame, torch::Tensor& buf) {
  // ref:
  // https://ffmpeg.org/doxygen/4.1/filtering__video_8c_source.html#l00179
  // https://ffmpeg.org/doxygen/4.1/decode__video_8c_source.html#l00038
  AVPixelFormat format = static_cast<AVPixelFormat>(frame->format);
  switch (format) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
    case AV_PIX_FMT_ARGB:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_ABGR:
    case AV_PIX_FMT_BGRA:
    case AV_PIX_FMT_GRAY8: {
      write_interlaced_image(frame, buf);
      return;
    }
    case AV_PIX_FMT_YUV444P: {
      write_planar_image(frame, buf);
      return;
    }
    case AV_PIX_FMT_YUV420P: {
      write_yuv420p(frame, buf);
      return;
    }
    case AV_PIX_FMT_NV12: {
      write_nv12_cpu(frame, buf);
      return;
    }
#ifdef USE_CUDA
    case AV_PIX_FMT_CUDA: {
      AVHWFramesContext* hwctx = (AVHWFramesContext*)frame->hw_frames_ctx->data;
      AVPixelFormat sw_format = hwctx->sw_format;
      // cuvid decoder (nvdec frontend of ffmpeg) only supports the following
      // output formats
      // https://github.com/FFmpeg/FFmpeg/blob/072101bd52f7f092ee976f4e6e41c19812ad32fd/libavcodec/cuviddec.c#L1121-L1124
      switch (sw_format) {
        case AV_PIX_FMT_NV12: {
          write_nv12_cuda(frame, buf);
          return;
        }
        case AV_PIX_FMT_P010:
        case AV_PIX_FMT_P016:
          TORCH_CHECK(
              false,
              "Unsupported video format found in CUDA HW: " +
                  std::string(av_get_pix_fmt_name(sw_format)));
        default:
          TORCH_CHECK(
              false,
              "Unexpected video format found in CUDA HW: " +
                  std::string(av_get_pix_fmt_name(sw_format)));
      }
    }
#endif
    default:
      TORCH_CHECK(
          false,
          "Unexpected video format: " +
              std::string(av_get_pix_fmt_name(format)));
  }
}

} // namespace

torch::Tensor convert_image(AVFrame* frame, const torch::Device& device) {
  auto [buffer, is_planar] = get_image_buffer(frame, 1, device);
  write_image(frame, buffer);
  return is_planar ? buffer : buffer.permute({0, 3, 1, 2});
}

} // namespace detail
} // namespace io
} // namespace torchaudio
