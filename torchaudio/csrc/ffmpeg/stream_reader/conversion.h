#pragma once
#include <torch/types.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio::io {

////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////
template <c10::ScalarType dtype, bool is_planar>
class AudioConverter {
  const int num_channels;

 public:
  AudioConverter(int num_channels);

  // Converts AVFrame* into Tensor of [T, C]
  torch::Tensor convert(const AVFrame* src);

  // Converts AVFrame* into pre-allocated Tensor.
  // The shape must be [C, T] if is_planar otherwise [T, C]
  void convert(const AVFrame* src, torch::Tensor& dst);
};

////////////////////////////////////////////////////////////////////////////////
// Image
////////////////////////////////////////////////////////////////////////////////
struct ImageConverterBase {
  const int height;
  const int width;
  const int num_channels;

  ImageConverterBase(int h, int w, int c);
};

////////////////////////////////////////////////////////////////////////////////
// Interlaced Images - NHWC
////////////////////////////////////////////////////////////////////////////////
struct InterlacedImageConverter : public ImageConverterBase {
  using ImageConverterBase::ImageConverterBase;
  // convert AVFrame* into Tensor of NCHW format
  torch::Tensor convert(const AVFrame* src);
  // convert AVFrame* into pre-allocated Tensor of NHWC format
  void convert(const AVFrame* src, torch::Tensor& dst);
};

struct Interlaced16BitImageConverter : public ImageConverterBase {
  using ImageConverterBase::ImageConverterBase;
  // convert AVFrame* into Tensor of NCHW format
  torch::Tensor convert(const AVFrame* src);
  // convert AVFrame* into pre-allocated Tensor of NHWC format
  void convert(const AVFrame* src, torch::Tensor& dst);
};

////////////////////////////////////////////////////////////////////////////////
// Planar Images - NCHW
////////////////////////////////////////////////////////////////////////////////
struct PlanarImageConverter : public ImageConverterBase {
  using ImageConverterBase::ImageConverterBase;
  void convert(const AVFrame* src, torch::Tensor& dst);
  torch::Tensor convert(const AVFrame* src);
};

////////////////////////////////////////////////////////////////////////////////
// Family of YUVs - NCHW
////////////////////////////////////////////////////////////////////////////////
class YUV420PConverter : public ImageConverterBase {
  torch::Tensor tmp_uv;

 public:
  YUV420PConverter(int height, int width);
  void convert(const AVFrame* src, torch::Tensor& dst);
  torch::Tensor convert(const AVFrame* src);
};

class NV12Converter : public ImageConverterBase {
  torch::Tensor tmp_uv;

 public:
  NV12Converter(int height, int width);
  void convert(const AVFrame* src, torch::Tensor& dst);
  torch::Tensor convert(const AVFrame* src);
};

#ifdef USE_CUDA

class NV12CudaConverter : ImageConverterBase {
  torch::Tensor tmp_uv;

 public:
  NV12CudaConverter(int height, int width, const torch::Device& device);
  void convert(const AVFrame* src, torch::Tensor& dst);
  torch::Tensor convert(const AVFrame* src);
};

class P010CudaConverter : ImageConverterBase {
  torch::Tensor tmp_uv;

 public:
  P010CudaConverter(int height, int width, const torch::Device& device);
  void convert(const AVFrame* src, torch::Tensor& dst);
  torch::Tensor convert(const AVFrame* src);
};

#endif
} // namespace torchaudio::io
