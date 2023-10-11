#include <libtorio/ffmpeg/stream_writer/packet_writer.h>

namespace torchaudio::io {
namespace {
AVStream* add_stream(
    AVFormatContext* format_ctx,
    const StreamParams& stream_params) {
  AVStream* stream = avformat_new_stream(format_ctx, nullptr);
  int ret =
      avcodec_parameters_copy(stream->codecpar, stream_params.codec_params);
  TORCH_CHECK(
      ret >= 0,
      "Failed to copy the stream's codec parameters. (",
      av_err2string(ret),
      ")");
  stream->time_base = stream_params.time_base;
  return stream;
}
} // namespace
PacketWriter::PacketWriter(
    AVFormatContext* format_ctx_,
    const StreamParams& stream_params_)
    : format_ctx(format_ctx_),
      stream(add_stream(format_ctx_, stream_params_)),
      original_time_base(stream_params_.time_base) {}

void PacketWriter::write_packet(const AVPacketPtr& packet) {
  AVPacket dst_packet;
  int ret = av_packet_ref(&dst_packet, packet);
  TORCH_CHECK(ret >= 0, "Failed to copy packet.");
  av_packet_rescale_ts(&dst_packet, original_time_base, stream->time_base);
  dst_packet.stream_index = stream->index;
  ret = av_interleaved_write_frame(format_ctx, &dst_packet);
  TORCH_CHECK(ret >= 0, "Failed to write packet to destination.");
}
} // namespace torchaudio::io
