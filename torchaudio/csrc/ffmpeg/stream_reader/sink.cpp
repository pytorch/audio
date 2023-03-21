#include <torchaudio/csrc/ffmpeg/hw_context.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/chunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/unchunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/sink.h>
#include <stdexcept>

namespace torchaudio {
namespace io {

namespace {
std::unique_ptr<Buffer> get_buffer(
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

  if (info.type == AVMEDIA_TYPE_AUDIO) {
    AVSampleFormat fmt = (AVSampleFormat)(info.format);
    if (frames_per_chunk == -1) {
      return detail::get_unchunked_buffer(
          info.time_base, fmt, info.num_channels);
    } else {
      return detail::get_chunked_buffer(
          info.time_base, frames_per_chunk, num_chunks, fmt, info.num_channels);
    }
  } else {
    AVPixelFormat fmt = (AVPixelFormat)(info.format);
    TORCH_INTERNAL_ASSERT(fmt != AV_PIX_FMT_CUDA);

    if (frames_per_chunk == -1) {
      return detail::get_unchunked_buffer(
          info.time_base, fmt, info.height, info.width, device);
    } else {
      return detail::get_chunked_buffer(
          info.time_base,
          frames_per_chunk,
          num_chunks,
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
  if (codec_ctx->hw_frames_ctx) {
    p.create_filter(av_buffer_ref(codec_ctx->hw_frames_ctx));
  } else {
    p.create_filter(nullptr);
  }
  return p;
}

} // namespace

Sink::Sink(
    AVRational input_time_base_,
    AVCodecContext* codec_ctx_,
    int frames_per_chunk,
    int num_chunks,
    AVRational frame_rate_,
    const std::string& filter_desc,
    const torch::Device& device)
    : input_time_base(input_time_base_),
      codec_ctx(codec_ctx_),
      frame_rate(frame_rate_),
      filter_description(filter_desc),
      filter(get_filter_graph(
          input_time_base_,
          codec_ctx,
          frame_rate,
          filter_description)),
      buffer(get_buffer(filter, frames_per_chunk, num_chunks, device)) {}

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
      buffer->push_frame(frame);
    }
    av_frame_unref(frame);
  }
  return ret;
}

void Sink::flush() {
  filter = get_filter_graph(
      input_time_base, codec_ctx, frame_rate, filter_description);
  buffer->flush();
}

} // namespace io
} // namespace torchaudio
