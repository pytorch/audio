#pragma once

#include <torch/types.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/filter_graph.h>

namespace torchaudio::io {

// Encoder + Muxer
class Encoder {
  // Reference to the AVFormatContext (muxer)
  AVFormatContext* format_ctx;
  // Reference to codec context (encoder)
  AVCodecContext* codec_ctx;
  // Stream object
  // Encoder object creates AVStream, but it will be deallocated along with
  // AVFormatContext, So Encoder does not own it.
  AVStream* stream;
  // Temporary object used during the encoding
  // Encoder owns it.
  AVPacketPtr packet{};

 public:
  Encoder(AVFormatContext* format_ctx, AVCodecContext* codec_ctx);

  void encode(AVFrame* frame);
};

} // namespace torchaudio::io
