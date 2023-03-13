#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/chunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/unchunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/sink.h>
#include <stdexcept>

namespace torchaudio {
namespace io {

namespace {
std::unique_ptr<Buffer> get_buffer(
    AVMediaType type,
    int frames_per_chunk,
    int num_chunks,
    double frame_duration,
    const torch::Device& device) {
  TORCH_CHECK(
      frames_per_chunk > 0 || frames_per_chunk == -1,
      "`frames_per_chunk` must be positive or -1. Found: ",
      frames_per_chunk);

  TORCH_CHECK(
      num_chunks > 0 || num_chunks == -1,
      "`num_chunks` must be positive or -1. Found: ",
      num_chunks);

  TORCH_INTERNAL_ASSERT(
      type == AVMEDIA_TYPE_AUDIO || type == AVMEDIA_TYPE_VIDEO,
      "Unsupported media type: ",
      av_get_media_type_string(type),
      ". Only video or audio is supported ");

  // Chunked Mode
  if (frames_per_chunk > 0) {
    if (type == AVMEDIA_TYPE_AUDIO) {
      return std::unique_ptr<Buffer>(new detail::ChunkedAudioBuffer(
          frames_per_chunk, num_chunks, frame_duration));
    } else {
      return std::unique_ptr<Buffer>(new detail::ChunkedVideoBuffer(
          frames_per_chunk, num_chunks, frame_duration, device));
    }
  } else { // unchunked mode
    if (type == AVMEDIA_TYPE_AUDIO) {
      return std::unique_ptr<Buffer>(new detail::UnchunkedAudioBuffer());
    } else {
      return std::unique_ptr<Buffer>(new detail::UnchunkedVideoBuffer(device));
    }
  }
}

std::unique_ptr<FilterGraph> get_filter_graph(
    AVRational input_time_base,
    AVCodecContext* codec_ctx,
    AVRational frame_rate,
    const std::string& filter_description) {
  auto p = std::make_unique<FilterGraph>(codec_ctx->codec_type);

  switch (codec_ctx->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
      p->add_audio_src(
          codec_ctx->sample_fmt,
          input_time_base,
          codec_ctx->sample_rate,
          codec_ctx->channel_layout);
      break;
    case AVMEDIA_TYPE_VIDEO:
      p->add_video_src(
          codec_ctx->pix_fmt,
          input_time_base,
          frame_rate,
          codec_ctx->width,
          codec_ctx->height,
          codec_ctx->sample_aspect_ratio);
      break;
    default:
      TORCH_CHECK(false, "Only audio/video are supported.");
  }
  p->add_sink();
  p->add_process(filter_description);
  p->create_filter();
  return p;
}

} // namespace

Sink::Sink(
    AVRational input_time_base_,
    AVCodecContext* codec_ctx_,
    int frames_per_chunk,
    int num_chunks,
    AVRational frame_rate_,
    const c10::optional<std::string>& filter_description_,
    const torch::Device& device)
    : input_time_base(input_time_base_),
      codec_ctx(codec_ctx_),
      frame_rate(frame_rate_),
      filter_description(filter_description_.value_or(
          codec_ctx->codec_type == AVMEDIA_TYPE_AUDIO ? "anull" : "null")),
      filter(get_filter_graph(
          input_time_base_,
          codec_ctx,
          frame_rate,
          filter_description)),
      output_time_base(filter->get_output_timebase()),
      buffer(get_buffer(
          codec_ctx->codec_type,
          frames_per_chunk,
          num_chunks,
          double(output_time_base.num) / output_time_base.den,
          device)) {}

// 0: some kind of success
// <0: Some error happened
int Sink::process_frame(AVFrame* pFrame) {
  int ret = filter->add_frame(pFrame);
  while (ret >= 0) {
    ret = filter->get_frame(frame);
    //  AVERROR(EAGAIN) means that new input data is required to return new
    //  output.
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      return 0;
    }
    if (ret >= 0) {
      double pts =
          double(frame->pts * output_time_base.num) / output_time_base.den;
      buffer->push_frame(frame, pts);
    }
    av_frame_unref(frame);
  }
  return ret;
}

std::string Sink::get_filter_description() const {
  return filter_description;
}

FilterGraphOutputInfo Sink::get_filter_output_info() const {
  return filter->get_output_info();
}

void Sink::flush() {
  filter = get_filter_graph(
      input_time_base, codec_ctx, frame_rate, filter_description);
  buffer->flush();
}

} // namespace io
} // namespace torchaudio
