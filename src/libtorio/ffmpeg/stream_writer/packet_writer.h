#pragma once
#include <libtorio/ffmpeg/ffmpeg.h>

namespace torchaudio::io {
class PacketWriter {
  AVFormatContext* format_ctx;
  AVStream* stream;
  AVRational original_time_base;

 public:
  PacketWriter(
      AVFormatContext* format_ctx_,
      const StreamParams& stream_params_);
  void write_packet(const AVPacketPtr& packet);
};
} // namespace torchaudio::io
