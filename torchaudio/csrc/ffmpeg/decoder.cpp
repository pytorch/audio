#include <torchaudio/csrc/ffmpeg/decoder.h>

namespace torchaudio {
namespace ffmpeg {

////////////////////////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////////////////////////
Decoder::Decoder(AVCodecParameters* pParam) : pCodecContext(pParam) {}

int Decoder::process_packet(AVPacket* pPacket) {
  return avcodec_send_packet(pCodecContext, pPacket);
}

int Decoder::get_frame(AVFrame* pFrame) {
  return avcodec_receive_frame(pCodecContext, pFrame);
}

} // namespace ffmpeg
} // namespace torchaudio
