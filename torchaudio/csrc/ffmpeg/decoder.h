#pragma once

#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace ffmpeg {

class Decoder {
  AVCodecContextPtr pCodecContext;

 public:
  // Default constructable
  Decoder(AVCodecParameters* pParam);
  // Custom destructor to clean up the resources
  ~Decoder() = default;
  // Non-copyable
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;
  // Movable
  Decoder(Decoder&&) = default;
  Decoder& operator=(Decoder&&) = default;

  // Process incoming packet
  int process_packet(AVPacket* pPacket);
  // Fetch a decoded frame
  int get_frame(AVFrame* pFrame);
};

} // namespace ffmpeg
} // namespace torchaudio
