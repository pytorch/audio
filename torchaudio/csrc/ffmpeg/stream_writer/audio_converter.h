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
  enum AVSampleFormat src_fmt;
  AVCodecContext* codec_ctx;
  AVFramePtr buffer;
  const int64_t buffer_size;
  SlicingTensorConverter::ConvertFunc convert_func;

 public:
  AudioTensorConverter(
      enum AVSampleFormat src_fmt,
      AVCodecContext* codec_ctx,
      int default_frame_size = 10000);
  SlicingTensorConverter convert(const torch::Tensor& frames);
};
} // namespace torchaudio::io
