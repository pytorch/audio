#include <torchaudio/csrc/ffmpeg/decoder.h>

namespace torchaudio {
namespace ffmpeg {

////////////////////////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////////////////////////
Decoder::Decoder(
    AVCodecParameters* pParam,
    const std::string& decoder_name,
    const std::map<std::string, std::string>& decoder_option,
    const torch::Device& device)
    : pCodecContext(get_decode_context(pParam->codec_id, decoder_name)) {
  init_codec_context(
      pCodecContext,
      pParam,
      decoder_name,
      decoder_option,
      device,
      pHWBufferRef);
}

int Decoder::process_packet(AVPacket* pPacket) {
  return avcodec_send_packet(pCodecContext, pPacket);
}

int Decoder::get_frame(AVFrame* pFrame) {
  return avcodec_receive_frame(pCodecContext, pFrame);
}

void Decoder::flush_buffer() {
  avcodec_flush_buffers(pCodecContext);
}

} // namespace ffmpeg
} // namespace torchaudio
