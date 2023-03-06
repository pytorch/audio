#pragma once

#include <torchaudio/csrc/ffmpeg/stream_writer/converter.h>

namespace torchaudio::io {

////////////////////////////////////////////////////////////////////////////////
// AudioTensorConverter
////////////////////////////////////////////////////////////////////////////////
// AudioTensorConverter is responsible for picking up the right set of
// conversion process (InitFunc and ConvertFunc) based on the input sample
// format information, and own them.
class AudioTensorConverter {
  AVFrame* buffer;
  const int64_t buffer_size;
  SlicingTensorConverter::ConvertFunc convert_func;

 public:
  AudioTensorConverter(AVFrame* buffer, const int64_t buffer_size);
  SlicingTensorConverter convert(const torch::Tensor& frames);
};
} // namespace torchaudio::io
