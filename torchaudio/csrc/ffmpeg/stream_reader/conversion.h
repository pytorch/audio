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
 public:
  YUV420PConverter(int height, int width);
  void convert(const AVFrame* src, torch::Tensor& dst);
  torch::Tensor convert(const AVFrame* src);
};

class YUV420P10LEConverter : public ImageConverterBase {
 public:
  YUV420P10LEConverter(int height, int width);
  void convert(const AVFrame* src, torch::Tensor& dst);
  torch::Tensor convert(const AVFrame* src);
};

class NV12Converter : public ImageConverterBase {
 public:
  NV12Converter(int height, int width);
  void convert(const AVFrame* src, torch::Tensor& dst);
  torch::Tensor convert(const AVFrame* src);
};

#ifdef USE_CUDA

// Note:
// GPU decoders are tricky. They allow to change the resolution as part of
// decoder option, and the resulting resolution is (seemingly) not retrievable.
// Therefore, we adopt delayed frame size initialization.
// For that purpose, we do not inherit from ImageConverterBase.
struct CudaImageConverterBase {
  const torch::Device device;
  bool init = false;
  int height = -1;
  int width = -1;
  explicit CudaImageConverterBase(const torch::Device& device);
};

class NV12CudaConverter : CudaImageConverterBase {
  torch::Tensor tmp_uv{};

 public:
  explicit NV12CudaConverter(const torch::Device& device);
  void convert(const AVFrame* src, torch::Tensor& dst);
  torch::Tensor convert(const AVFrame* src);
};

class P010CudaConverter : CudaImageConverterBase {
  torch::Tensor tmp_uv{};

 public:
  explicit P010CudaConverter(const torch::Device& device);
  void convert(const AVFrame* src, torch::Tensor& dst);
  torch::Tensor convert(const AVFrame* src);
};

class YUV444PCudaConverter : CudaImageConverterBase {
 public:
  explicit YUV444PCudaConverter(const torch::Device& device);
  void convert(const AVFrame* src, torch::Tensor& dst);
  torch::Tensor convert(const AVFrame* src);
};

#endif
} // namespace torchaudio::io
