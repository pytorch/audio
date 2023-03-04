#include <torchaudio/csrc/ffmpeg/stream_writer/audio_output_stream.h>

namespace torchaudio::io {

namespace {

FilterGraph get_audio_filter(
    AVSampleFormat src_fmt,
    AVCodecContext* codec_ctx) {
  auto desc = [&]() -> std::string {
    if (src_fmt == codec_ctx->sample_fmt) {
      return "anull";
    } else {
      std::stringstream ss;
      ss << "aformat=" << av_get_sample_fmt_name(codec_ctx->sample_fmt);
      return ss.str();
    }
  }();

  FilterGraph p{AVMEDIA_TYPE_AUDIO};
  p.add_audio_src(
      src_fmt,
      codec_ctx->time_base,
      codec_ctx->sample_rate,
      codec_ctx->channel_layout);
  p.add_sink();
  p.add_process(desc);
  p.create_filter();
  return p;
}

AVFramePtr get_audio_frame(
    AVSampleFormat src_fmt,
    AVCodecContext* codec_ctx,
    int default_frame_size = 10000) {
  AVFramePtr frame{};
  frame->pts = 0;
  frame->format = src_fmt;
  frame->channel_layout = codec_ctx->channel_layout;
  frame->sample_rate = codec_ctx->sample_rate;
  frame->nb_samples =
      codec_ctx->frame_size ? codec_ctx->frame_size : default_frame_size;
  if (frame->nb_samples) {
    int ret = av_frame_get_buffer(frame, 0);
    TORCH_CHECK(
        ret >= 0,
        "Error allocating an audio buffer (",
        av_err2string(ret),
        ").");
  }
  return frame;
}

} // namespace

AudioOutputStream::AudioOutputStream(
    AVFormatContext* format_ctx,
    AVSampleFormat src_fmt,
    AVCodecContextPtr&& codec_ctx_)
    : OutputStream(
          format_ctx,
          codec_ctx_,
          get_audio_filter(src_fmt, codec_ctx_)),
      buffer(get_audio_frame(src_fmt, codec_ctx_)),
      converter(buffer, buffer->nb_samples),
      codec_ctx(std::move(codec_ctx_)) {}

void AudioOutputStream::write_chunk(const torch::Tensor& waveform) {
  AVRational time_base{1, codec_ctx->sample_rate};
  for (const auto& frame : converter.convert(waveform)) {
    process_frame(frame);
    frame->pts +=
        av_rescale_q(frame->nb_samples, time_base, codec_ctx->time_base);
  }
}

} // namespace torchaudio::io
