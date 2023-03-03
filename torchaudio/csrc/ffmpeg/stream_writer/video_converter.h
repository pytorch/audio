#pragma once

#include <torchaudio/csrc/ffmpeg/stream_writer/converter.h>

namespace torchaudio::io {

////////////////////////////////////////////////////////////////////////////////
// VideoTensorConverter
////////////////////////////////////////////////////////////////////////////////
// VideoTensorConverter is responsible for picking up the right set of
// conversion process (InitFunc and ConvertFunc) based on the input pixel format
// information, and own them.
class VideoTensorConverter {
 public:
  // Initialization is one-time process applied to frames before the iteration
  // starts. i.e. either convert to NHWC.
  using InitFunc = std::function<torch::Tensor(const torch::Tensor&)>;

 private:
  AVFrame* buffer;

  InitFunc init_func{};
  SlicingTensorConverter::ConvertFunc convert_func{};

 public:
  explicit VideoTensorConverter(AVFrame* buffer);
  SlicingTensorConverter convert(const torch::Tensor& frames);
};
} // namespace torchaudio::io
