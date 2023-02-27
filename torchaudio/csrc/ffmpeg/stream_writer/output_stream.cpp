#include <torchaudio/csrc/ffmpeg/stream_writer/output_stream.h>

namespace torchaudio::io {

OutputStream::OutputStream(
    AVFormatContext* format_ctx,
    AVCodecContext* codec_ctx_,
    std::unique_ptr<FilterGraph>&& filter_)
    : codec_ctx(codec_ctx_),
      encoder(format_ctx, codec_ctx),
      filter(std::move(filter_)),
      dst_frame(),
      num_frames(0) {}

void OutputStream::process_frame(AVFrame* src) {
  if (!filter) {
    encoder.encode(src);
    return;
  }
  int ret = filter->add_frame(src);
  while (ret >= 0) {
    ret = filter->get_frame(dst_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      if (ret == AVERROR_EOF) {
        encoder.encode(nullptr);
      }
      break;
    }
    if (ret >= 0) {
      encoder.encode(dst_frame);
    }
    av_frame_unref(dst_frame);
  }
}

void OutputStream::flush() {
  process_frame(nullptr);
}

} // namespace torchaudio::io
