#pragma once

#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace ffmpeg {

class Decoder {
  AVCodecContextPtr pCodecContext;

 public:
  // Default constructable
  Decoder(
      AVCodecParameters* pParam,
      const std::string& decoder_name,
      const std::map<std::string, std::string>& decoder_option);
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
  // Flush buffer (for seek)
  void flush_buffer();
};

} // namespace ffmpeg
} // namespace torchaudio
