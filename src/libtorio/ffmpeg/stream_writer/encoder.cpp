#include <libtorio/ffmpeg/stream_writer/encoder.h>

namespace torio::io {

Encoder::Encoder(
    AVFormatContext* format_ctx,
    AVCodecContext* codec_ctx,
    AVStream* stream) noexcept
    : format_ctx(format_ctx), codec_ctx(codec_ctx), stream(stream) {}

///
/// Encode the given AVFrame data
///
/// @param frame Frame data to encode
void Encoder::encode(AVFrame* frame) {
  int ret = avcodec_send_frame(codec_ctx, frame);
  TORCH_CHECK(ret >= 0, "Failed to encode frame (", av_err2string(ret), ").");
  while (ret >= 0) {
    ret = avcodec_receive_packet(codec_ctx, packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      if (ret == AVERROR_EOF) {
        // Note:
        // av_interleaved_write_frame buffers the packets internally as needed
        // to make sure the packets in the output file are properly interleaved
        // in the order of increasing dts.
        // https://ffmpeg.org/doxygen/3.4/group__lavf__encoding.html#ga37352ed2c63493c38219d935e71db6c1
        // Passing nullptr will (forcefully) flush the queue, and this is
        // necessary if users mal-configure the streams.

        // Possible follow up: Add flush_buffer method?
        // An alternative is to use `av_write_frame` functoin, but in that case
        // client code is responsible for ordering packets, which makes it
        // complicated to use StreamingMediaEncoder
        ret = av_interleaved_write_frame(format_ctx, nullptr);
        TORCH_CHECK(
            ret >= 0, "Failed to flush packet (", av_err2string(ret), ").");
      }
      break;
    } else {
      TORCH_CHECK(
          ret >= 0,
          "Failed to fetch encoded packet (",
          av_err2string(ret),
          ").");
    }
    // https://github.com/pytorch/audio/issues/2790
    // If this is not set, the last frame is not properly saved, as
    // the encoder cannot figure out when the packet should finish.
    if (packet->duration == 0 && codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
      // 1 means that 1 frame (in codec time base, which is the frame rate)
      // This has to be set before av_packet_rescale_ts bellow.
      packet->duration = 1;
    }
    av_packet_rescale_ts(packet, codec_ctx->time_base, stream->time_base);
    packet->stream_index = stream->index;

    ret = av_interleaved_write_frame(format_ctx, packet);
    TORCH_CHECK(ret >= 0, "Failed to write packet (", av_err2string(ret), ").");
  }
}

} // namespace torio::io
