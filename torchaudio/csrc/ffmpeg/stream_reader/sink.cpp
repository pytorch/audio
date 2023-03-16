#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/chunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/unchunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/sink.h>
#include <stdexcept>

namespace torchaudio {
namespace io {

namespace {
std::unique_ptr<Buffer> get_buffer(
    AVCodecContext* codec_ctx,
    FilterGraph& filter,
    int frames_per_chunk,
    int num_chunks,
    const torch::Device& device) {
  TORCH_CHECK(
      frames_per_chunk > 0 || frames_per_chunk == -1,
      "`frames_per_chunk` must be positive or -1. Found: ",
      frames_per_chunk);

  TORCH_CHECK(
      num_chunks > 0 || num_chunks == -1,
      "`num_chunks` must be positive or -1. Found: ",
      num_chunks);

  auto info = filter.get_output_info();

  TORCH_CHECK(
      info.type == AVMEDIA_TYPE_AUDIO || info.type == AVMEDIA_TYPE_VIDEO,
      "Unsupported media type: ",
      av_get_media_type_string(info.type),
      ". Only video or audio is supported ");

  auto time_base = filter.get_output_timebase();
  double frame_duration = double(time_base.num) / time_base.den;

  if (info.type == AVMEDIA_TYPE_AUDIO) {
    AVSampleFormat fmt = (AVSampleFormat)(info.format);
    if (frames_per_chunk == -1) {
      return detail::get_unchunked_buffer(fmt, codec_ctx->channels);
    } else {
      return detail::get_chunked_buffer(
          frames_per_chunk,
          num_chunks,
          frame_duration,
          fmt,
          codec_ctx->channels);
    }
  } else {
    // Note
    // When using HW decoder, the pixel format is CUDA, and FilterGraph does
    // not yet support CUDA frames, nor propagating the software pixel format,
    // so here, we refer to AVCodecContext* to look at the pixel format.
    AVPixelFormat fmt = (AVPixelFormat)(info.format);
    if (fmt == AV_PIX_FMT_CUDA) {
      fmt = codec_ctx->sw_pix_fmt;
    }

    if (frames_per_chunk == -1) {
      return detail::get_unchunked_buffer(fmt, info.height, info.width, device);
    } else {
      return detail::get_chunked_buffer(
          frames_per_chunk,
          num_chunks,
          frame_duration,
          fmt,
          info.height,
          info.width,
          device);
    }
  }
}

FilterGraph get_filter_graph(
    AVRational input_time_base,
    AVCodecContext* codec_ctx,
    AVRational frame_rate,
    const std::string& filter_description) {
  auto p = FilterGraph{codec_ctx->codec_type};

  switch (codec_ctx->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
      p.add_audio_src(
          codec_ctx->sample_fmt,
          input_time_base,
          codec_ctx->sample_rate,
          codec_ctx->channel_layout);
      break;
    case AVMEDIA_TYPE_VIDEO:
      p.add_video_src(
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
  p.add_sink();
  p.add_process(filter_description);
  p.create_filter();
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
      output_time_base(filter.get_output_timebase()),
      buffer(
          get_buffer(codec_ctx, filter, frames_per_chunk, num_chunks, device)) {
}

// 0: some kind of success
// <0: Some error happened
int Sink::process_frame(AVFrame* pFrame) {
  int ret = filter.add_frame(pFrame);
  while (ret >= 0) {
    ret = filter.get_frame(frame);
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
  return filter.get_output_info();
}

void Sink::flush() {
  filter = get_filter_graph(
      input_time_base, codec_ctx, frame_rate, filter_description);
  buffer->flush();
}

} // namespace io
} // namespace torchaudio
