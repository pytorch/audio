#pragma once

#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace ffmpeg {

class Decoder {
  AVCodecContextPtr pCodecContext;
  enum AVPixelFormat pHwFmt = AV_PIX_FMT_NONE;

 public:
  // Default constructable
  Decoder(
      AVCodecParameters* pParam,
      const c10::optional<std::string>& decoder_name,
      const c10::optional<OptionDict>& decoder_option,
      const torch::Device& device);
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
