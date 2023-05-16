#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/conversion.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace torchaudio::io {

////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////

template <c10::ScalarType dtype, bool is_planar>
AudioConverter<dtype, is_planar>::AudioConverter(int c) : num_channels(c) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_channels > 0);
}

template <c10::ScalarType dtype, bool is_planar>
torch::Tensor AudioConverter<dtype, is_planar>::convert(const AVFrame* src) {
  if constexpr (is_planar) {
    torch::Tensor dst = torch::empty({num_channels, src->nb_samples}, dtype);
    convert(src, dst);
    return dst.permute({1, 0});
  } else {
    torch::Tensor dst = torch::empty({src->nb_samples, num_channels}, dtype);
    convert(src, dst);
    return dst;
  }
}

// Converts AVFrame* into pre-allocated Tensor.
// The shape must be [C, T] if is_planar otherwise [T, C]
template <c10::ScalarType dtype, bool is_planar>
void AudioConverter<dtype, is_planar>::convert(
    const AVFrame* src,
    torch::Tensor& dst) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_channels == src->channels);

  constexpr int bps = []() {
    switch (dtype) {
      case torch::kUInt8:
        return 1;
      case torch::kInt16:
        return 2;
      case torch::kInt32:
      case torch::kFloat32:
        return 4;
      case torch::kInt64:
      case torch::kFloat64:
        return 8;
    }
  }();

  // Note
  // FFMpeg's `nb_samples` represnts the number of samples par channel.
  // whereas, in torchaudio, `num_samples` is used to represent the number of
  // samples across channels. torchaudio uses `num_frames` for per-channel
  // samples.
  if constexpr (is_planar) {
    int plane_size = bps * src->nb_samples;
    uint8_t* p_dst = static_cast<uint8_t*>(dst.data_ptr());
    for (int i = 0; i < num_channels; ++i) {
      memcpy(p_dst, src->extended_data[i], plane_size);
      p_dst += plane_size;
    }
  } else {
    int plane_size = bps * src->nb_samples * num_channels;
    memcpy(dst.data_ptr(), src->extended_data[0], plane_size);
  }
}

// Explicit instantiation
template class AudioConverter<torch::kUInt8, false>;
template class AudioConverter<torch::kUInt8, true>;
template class AudioConverter<torch::kInt16, false>;
template class AudioConverter<torch::kInt16, true>;
template class AudioConverter<torch::kInt32, false>;
template class AudioConverter<torch::kInt32, true>;
template class AudioConverter<torch::kInt64, false>;
template class AudioConverter<torch::kInt64, true>;
template class AudioConverter<torch::kFloat32, false>;
template class AudioConverter<torch::kFloat32, true>;
template class AudioConverter<torch::kFloat64, false>;
template class AudioConverter<torch::kFloat64, true>;

////////////////////////////////////////////////////////////////////////////////
// Image
////////////////////////////////////////////////////////////////////////////////

namespace {

torch::Tensor get_image_buffer(
    at::IntArrayRef shape,
    const torch::Dtype dtype = torch::kUInt8) {
  return torch::empty(
      shape, torch::TensorOptions().dtype(dtype).layout(torch::kStrided));
}

torch::Tensor get_image_buffer(
    at::IntArrayRef shape,
    torch::Device device,
    const torch::Dtype dtype = torch::kUInt8) {
  return torch::empty(
      shape,
      torch::TensorOptions()
          .dtype(dtype)
          .layout(torch::kStrided)
          .device(device));
}

} // namespace

ImageConverterBase::ImageConverterBase(int h, int w, int c)
    : height(h), width(w), num_channels(c) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(height > 0);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(width > 0);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_channels > 0);
}

////////////////////////////////////////////////////////////////////////////////
// Interlaced Image
////////////////////////////////////////////////////////////////////////////////
void InterlacedImageConverter::convert(const AVFrame* src, torch::Tensor& dst) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(1) == height);
  int stride = width * num_channels;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(2) * dst.size(3) == stride);
  auto p_dst = dst.data_ptr<uint8_t>();
  uint8_t* p_src = src->data[0];
  for (int i = 0; i < height; ++i) {
    memcpy(p_dst, p_src, stride);
    p_src += src->linesize[0];
    p_dst += stride;
  }
}

torch::Tensor InterlacedImageConverter::convert(const AVFrame* src) {
  torch::Tensor buffer = get_image_buffer({1, height, width, num_channels});
  convert(src, buffer);
  return buffer.permute({0, 3, 1, 2});
}

////////////////////////////////////////////////////////////////////////////////
// Interlaced 16 Bit Image
////////////////////////////////////////////////////////////////////////////////
void Interlaced16BitImageConverter::convert(
    const AVFrame* src,
    torch::Tensor& dst) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(1) == height);
  int stride = width * num_channels;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(2) * dst.size(3) == stride);
  auto p_dst = dst.data_ptr<int16_t>();
  uint8_t* p_src = src->data[0];
  for (int i = 0; i < height; ++i) {
    memcpy(p_dst, p_src, stride * 2);
    p_src += src->linesize[0];
    p_dst += stride;
  }
  // correct for int16
  dst += 32768;
}

torch::Tensor Interlaced16BitImageConverter::convert(const AVFrame* src) {
  torch::Tensor buffer =
      get_image_buffer({1, height, width, num_channels}, torch::kInt16);
  convert(src, buffer);
  return buffer.permute({0, 3, 1, 2});
}

////////////////////////////////////////////////////////////////////////////////
// Planar Image
////////////////////////////////////////////////////////////////////////////////
void PlanarImageConverter::convert(const AVFrame* src, torch::Tensor& dst) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->width == width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(1) == num_channels);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(2) == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(3) == width);

  for (int i = 0; i < num_channels; ++i) {
    torch::Tensor plane = dst.index({0, i});
    uint8_t* p_dst = plane.data_ptr<uint8_t>();
    uint8_t* p_src = src->data[i];
    int linesize = src->linesize[i];
    for (int h = 0; h < height; ++h) {
      memcpy(p_dst, p_src, width);
      p_src += linesize;
      p_dst += width;
    }
  }
}

torch::Tensor PlanarImageConverter::convert(const AVFrame* src) {
  torch::Tensor buffer = get_image_buffer({1, num_channels, height, width});
  convert(src, buffer);
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// YUV420P
////////////////////////////////////////////////////////////////////////////////
YUV420PConverter::YUV420PConverter(int h, int w)
    : ImageConverterBase(h, w, 3),
      tmp_uv(get_image_buffer({1, 2, height / 2, width / 2})) {
  TORCH_WARN_ONCE(
      "The output format YUV420P is selected. "
      "This will be implicitly converted to YUV444P, "
      "in which all the color components Y, U, V have the same dimension.");
}

void YUV420PConverter::convert(const AVFrame* src, torch::Tensor& dst) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      (AVPixelFormat)(src->format) == AV_PIX_FMT_YUV420P);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->width == width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(1) == 3);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(2) == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(3) == width);

  // Write Y plane directly
  {
    uint8_t* p_dst = dst.data_ptr<uint8_t>();
    uint8_t* p_src = src->data[0];
    for (int h = 0; h < height; ++h) {
      memcpy(p_dst, p_src, width);
      p_dst += width;
      p_src += src->linesize[0];
    }
  }
  // Write intermediate UV plane
  {
    uint8_t* p_dst = tmp_uv.data_ptr<uint8_t>();
    uint8_t* p_src = src->data[1];
    for (int h = 0; h < height / 2; ++h) {
      memcpy(p_dst, p_src, width / 2);
      p_dst += width / 2;
      p_src += src->linesize[1];
    }
    p_src = src->data[2];
    for (int h = 0; h < height / 2; ++h) {
      memcpy(p_dst, p_src, width / 2);
      p_dst += width / 2;
      p_src += src->linesize[2];
    }
  }
  // Upsample width and height
  namespace F = torch::nn::functional;
  torch::Tensor uv = F::interpolate(
      tmp_uv,
      F::InterpolateFuncOptions()
          .mode(torch::kNearest)
          .size(std::vector<int64_t>({height, width})));
  // Write to the UV plane
  // dst[:, 1:] = uv
  using namespace torch::indexing;
  dst.index_put_({Slice(), Slice(1)}, uv);
}

torch::Tensor YUV420PConverter::convert(const AVFrame* src) {
  torch::Tensor buffer = get_image_buffer({1, num_channels, height, width});
  convert(src, buffer);
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// YUV420P10LE
////////////////////////////////////////////////////////////////////////////////
YUV420P10LEConverter::YUV420P10LEConverter(int h, int w)
    : ImageConverterBase(h, w, 3) {
  TORCH_WARN_ONCE(
      "The output format YUV420PLE is selected. "
      "This will be implicitly converted to YUV444P (16-bit), "
      "in which all the color components Y, U, V have the same dimension.");
}

void YUV420P10LEConverter::convert(const AVFrame* src, torch::Tensor& dst) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      (AVPixelFormat)(src->format) == AV_PIX_FMT_YUV420P10LE);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->width == width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(1) == 3);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(2) == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(3) == width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.dtype() == torch::kInt16);

  // Write Y plane directly
  {
    int16_t* p_dst = dst.data_ptr<int16_t>();
    uint8_t* p_src = src->data[0];
    for (int h = 0; h < height; ++h) {
      memcpy(p_dst, p_src, (size_t)width * 2);
      p_dst += width;
      p_src += src->linesize[0];
    }
  }
  // Chroma (U and V planes) are subsamapled by 2 in both vertical and
  // holizontal directions.
  // https://en.wikipedia.org/wiki/Chroma_subsampling
  // Since we are returning data in Tensor, which has the same size for all
  // color planes, we need to upsample the UV planes. PyTorch has interpolate
  // function but it does not work for int16 type. So we manually copy them.
  //
  //              block1  block2  block3  block4
  // ab -> aabb = a  b   *  a  b *       *
  // cd    aabb                   a  b      a  b
  //       ccdd   c  d      c  d
  //       ccdd                   c  d      c  d
  //
  auto block00 = dst.slice(2, 0, {}, 2).slice(3, 0, {}, 2);
  auto block01 = dst.slice(2, 0, {}, 2).slice(3, 1, {}, 2);
  auto block10 = dst.slice(2, 1, {}, 2).slice(3, 0, {}, 2);
  auto block11 = dst.slice(2, 1, {}, 2).slice(3, 1, {}, 2);
  for (int i = 1; i < 3; ++i) {
    // borrow data
    auto tmp = torch::from_blob(
        src->data[i],
        {height / 2, width / 2},
        {src->linesize[i] / 2, 1},
        [](void*) {},
        torch::TensorOptions().dtype(torch::kInt16).layout(torch::kStrided));
    // Copy to each block
    block00.slice(1, i, i + 1).copy_(tmp);
    block01.slice(1, i, i + 1).copy_(tmp);
    block10.slice(1, i, i + 1).copy_(tmp);
    block11.slice(1, i, i + 1).copy_(tmp);
  }
}

torch::Tensor YUV420P10LEConverter::convert(const AVFrame* src) {
  torch::Tensor buffer =
      get_image_buffer({1, num_channels, height, width}, torch::kInt16);
  convert(src, buffer);
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// NV12
////////////////////////////////////////////////////////////////////////////////
NV12Converter::NV12Converter(int h, int w)
    : ImageConverterBase(h, w, 3),
      tmp_uv(get_image_buffer({1, height / 2, width / 2, 2})) {
  TORCH_WARN_ONCE(
      "The output format NV12 is selected. "
      "This will be implicitly converted to YUV444P, "
      "in which all the color components Y, U, V have the same dimension.");
}

void NV12Converter::convert(const AVFrame* src, torch::Tensor& dst) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      (AVPixelFormat)(src->format) == AV_PIX_FMT_NV12);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->width == width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(1) == 3);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(2) == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(3) == width);

  // Write Y plane directly
  {
    uint8_t* p_dst = dst.data_ptr<uint8_t>();
    uint8_t* p_src = src->data[0];
    for (int h = 0; h < height; ++h) {
      memcpy(p_dst, p_src, width);
      p_dst += width;
      p_src += src->linesize[0];
    }
  }
  // Write intermediate UV plane
  {
    uint8_t* p_dst = tmp_uv.data_ptr<uint8_t>();
    uint8_t* p_src = src->data[1];
    for (int h = 0; h < height / 2; ++h) {
      memcpy(p_dst, p_src, width);
      p_dst += width;
      p_src += src->linesize[1];
    }
  }
  // Upsample width and height
  namespace F = torch::nn::functional;
  torch::Tensor uv = F::interpolate(
      tmp_uv.permute({0, 3, 1, 2}),
      F::InterpolateFuncOptions()
          .mode(torch::kNearest)
          .size(std::vector<int64_t>({height, width})));

  // Write to the UV plane
  // dst[:, 1:] = uv
  using namespace torch::indexing;
  dst.index_put_({Slice(), Slice(1)}, uv);
}

torch::Tensor NV12Converter::convert(const AVFrame* src) {
  torch::Tensor buffer = get_image_buffer({1, num_channels, height, width});
  convert(src, buffer);
  return buffer;
}

#ifdef USE_CUDA

////////////////////////////////////////////////////////////////////////////////
// NV12 CUDA
////////////////////////////////////////////////////////////////////////////////
NV12CudaConverter::NV12CudaConverter(int h, int w, const torch::Device& device)
    : ImageConverterBase(h, w, 3),
      tmp_uv(get_image_buffer(
          {1, height / 2, width / 2, 2},
          device,
          torch::kUInt8)) {
  TORCH_WARN_ONCE(
      "The output format NV12 is selected. "
      "This will be implicitly converted to YUV444P, "
      "in which all the color components Y, U, V have the same dimension.");
}

void NV12CudaConverter::convert(const AVFrame* src, torch::Tensor& dst) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->width == width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(1) == 3);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(2) == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(3) == width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.dtype() == torch::kUInt8);

  auto fmt = (AVPixelFormat)(src->format);
  AVHWFramesContext* hwctx = (AVHWFramesContext*)src->hw_frames_ctx->data;
  AVPixelFormat sw_fmt = hwctx->sw_format;

  TORCH_INTERNAL_ASSERT(
      AV_PIX_FMT_CUDA == fmt,
      "Expected CUDA frame. Found: ",
      av_get_pix_fmt_name(fmt));
  TORCH_INTERNAL_ASSERT(
      AV_PIX_FMT_NV12 == sw_fmt,
      "Expected NV12 format. Found: ",
      av_get_pix_fmt_name(sw_fmt));

  // Write Y plane directly
  auto status = cudaMemcpy2D(
      dst.data_ptr(),
      width,
      src->data[0],
      src->linesize[0],
      width,
      height,
      cudaMemcpyDeviceToDevice);
  TORCH_CHECK(cudaSuccess == status, "Failed to copy Y plane to Cuda tensor.");
  // Preapare intermediate UV planes
  status = cudaMemcpy2D(
      tmp_uv.data_ptr(),
      width,
      src->data[1],
      src->linesize[1],
      width,
      height / 2,
      cudaMemcpyDeviceToDevice);
  TORCH_CHECK(cudaSuccess == status, "Failed to copy UV plane to Cuda tensor.");
  // Upsample width and height
  namespace F = torch::nn::functional;
  torch::Tensor uv = F::interpolate(
      tmp_uv.permute({0, 3, 1, 2}),
      F::InterpolateFuncOptions()
          .mode(torch::kNearest)
          .size(std::vector<int64_t>({height, width})));
  // Write to the UV plane
  // dst[:, 1:] = uv
  using namespace torch::indexing;
  dst.index_put_({Slice(), Slice(1)}, uv);
}

torch::Tensor NV12CudaConverter::convert(const AVFrame* src) {
  torch::Tensor buffer =
      get_image_buffer({1, num_channels, height, width}, tmp_uv.device());
  convert(src, buffer);
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// P010 CUDA
////////////////////////////////////////////////////////////////////////////////
P010CudaConverter::P010CudaConverter(int h, int w, const torch::Device& device)
    : ImageConverterBase(h, w, 3),
      tmp_uv(get_image_buffer(
          {1, height / 2, width / 2, 2},
          device,
          torch::kInt16)) {
  TORCH_WARN_ONCE(
      "The output format P010 is selected. "
      "This will be implicitly converted to YUV444P, "
      "in which all the color components Y, U, V have the same dimension.");
}

void P010CudaConverter::convert(const AVFrame* src, torch::Tensor& dst) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->width == width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(1) == 3);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(2) == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(3) == width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.dtype() == torch::kInt16);

  auto fmt = (AVPixelFormat)(src->format);
  AVHWFramesContext* hwctx = (AVHWFramesContext*)src->hw_frames_ctx->data;
  AVPixelFormat sw_fmt = hwctx->sw_format;

  TORCH_INTERNAL_ASSERT(
      AV_PIX_FMT_CUDA == fmt,
      "Expected CUDA frame. Found: ",
      av_get_pix_fmt_name(fmt));
  TORCH_INTERNAL_ASSERT(
      AV_PIX_FMT_P010 == sw_fmt,
      "Expected P010 format. Found: ",
      av_get_pix_fmt_name(sw_fmt));

  // Write Y plane directly
  auto status = cudaMemcpy2D(
      dst.data_ptr(),
      width * 2,
      src->data[0],
      src->linesize[0],
      width * 2,
      height,
      cudaMemcpyDeviceToDevice);
  TORCH_CHECK(cudaSuccess == status, "Failed to copy Y plane to CUDA tensor.");
  // Prepare intermediate UV planes
  status = cudaMemcpy2D(
      tmp_uv.data_ptr(),
      width * 2,
      src->data[1],
      src->linesize[1],
      width * 2,
      height / 2,
      cudaMemcpyDeviceToDevice);
  TORCH_CHECK(cudaSuccess == status, "Failed to copy UV plane to CUDA tensor.");
  // Write to the UV plane
  torch::Tensor uv = tmp_uv.permute({0, 3, 1, 2});
  using namespace torch::indexing;
  // very simplistic upscale using indexing since interpolate doesn't support
  // shorts
  dst.index_put_(
      {Slice(), Slice(1, 3), Slice(None, None, 2), Slice(None, None, 2)}, uv);
  dst.index_put_(
      {Slice(), Slice(1, 3), Slice(1, None, 2), Slice(None, None, 2)}, uv);
  dst.index_put_(
      {Slice(), Slice(1, 3), Slice(None, None, 2), Slice(1, None, 2)}, uv);
  dst.index_put_(
      {Slice(), Slice(1, 3), Slice(1, None, 2), Slice(1, None, 2)}, uv);
  // correct for int16
  dst += 32768;
}

torch::Tensor P010CudaConverter::convert(const AVFrame* src) {
  torch::Tensor buffer = get_image_buffer(
      {1, num_channels, height, width}, tmp_uv.device(), torch::kInt16);
  convert(src, buffer);
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// YUV444P CUDA
////////////////////////////////////////////////////////////////////////////////
YUV444PCudaConverter::YUV444PCudaConverter(
    int h,
    int w,
    const torch::Device& device)
    : ImageConverterBase(h, w, 3), device(device) {}

void YUV444PCudaConverter::convert(const AVFrame* src, torch::Tensor& dst) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src->width == width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(1) == 3);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(2) == height);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.size(3) == width);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dst.dtype() == torch::kUInt8);

  auto fmt = (AVPixelFormat)(src->format);
  AVHWFramesContext* hwctx = (AVHWFramesContext*)src->hw_frames_ctx->data;
  AVPixelFormat sw_fmt = hwctx->sw_format;

  TORCH_INTERNAL_ASSERT(
      AV_PIX_FMT_CUDA == fmt,
      "Expected CUDA frame. Found: ",
      av_get_pix_fmt_name(fmt));
  TORCH_INTERNAL_ASSERT(
      AV_PIX_FMT_YUV444P == sw_fmt,
      "Expected YUV444P format. Found: ",
      av_get_pix_fmt_name(sw_fmt));

  // Write Y plane directly
  for (int i = 0; i < num_channels; ++i) {
    auto status = cudaMemcpy2D(
        dst.index({0, i}).data_ptr(),
        width,
        src->data[i],
        src->linesize[i],
        width,
        height,
        cudaMemcpyDeviceToDevice);
    TORCH_CHECK(
        cudaSuccess == status, "Failed to copy plane ", i, " to CUDA tensor.");
  }
}

torch::Tensor YUV444PCudaConverter::convert(const AVFrame* src) {
  torch::Tensor buffer =
      get_image_buffer({1, num_channels, height, width}, device);
  convert(src, buffer);
  return buffer;
}

#endif

} // namespace torchaudio::io
