#include <torchaudio/csrc/ffmpeg/stream_writer/video_output_stream.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace torchaudio::io {

namespace {

FilterGraph get_video_filter(AVPixelFormat src_fmt, AVCodecContext* codec_ctx) {
  auto desc = [&]() -> std::string {
    if (src_fmt == codec_ctx->pix_fmt ||
        codec_ctx->pix_fmt == AV_PIX_FMT_CUDA) {
      return "null";
    } else {
      std::stringstream ss;
      ss << "format=" << av_get_pix_fmt_name(codec_ctx->pix_fmt);
      return ss.str();
    }
  }();

  FilterGraph p{AVMEDIA_TYPE_VIDEO};
  p.add_video_src(
      src_fmt,
      codec_ctx->time_base,
      codec_ctx->width,
      codec_ctx->height,
      codec_ctx->sample_aspect_ratio);
  p.add_sink();
  p.add_process(desc);
  p.create_filter();
  return p;
}

AVFramePtr get_video_frame(AVPixelFormat src_fmt, AVCodecContext* codec_ctx) {
  AVFramePtr frame{};
  if (codec_ctx->hw_frames_ctx) {
    int ret = av_hwframe_get_buffer(codec_ctx->hw_frames_ctx, frame, 0);
    TORCH_CHECK(ret >= 0, "Failed to fetch CUDA frame: ", av_err2string(ret));
  } else {
    frame->format = src_fmt;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;

    int ret = av_frame_get_buffer(frame, 0);
    TORCH_CHECK(
        ret >= 0,
        "Error allocating a video buffer (",
        av_err2string(ret),
        ").");
  }
  frame->pts = 0;
  return frame;
}

} // namespace

VideoOutputStream::VideoOutputStream(
    AVFormatContext* format_ctx,
    AVPixelFormat src_fmt,
    AVCodecContextPtr&& codec_ctx_,
    AVBufferRefPtr&& hw_device_ctx_,
    AVBufferRefPtr&& hw_frame_ctx_)
    : OutputStream(
          format_ctx,
          codec_ctx_,
          get_video_filter(src_fmt, codec_ctx_)),
      buffer(get_video_frame(src_fmt, codec_ctx_)),
      converter(buffer),
      hw_device_ctx(std::move(hw_device_ctx_)),
      hw_frame_ctx(std::move(hw_frame_ctx_)),
      codec_ctx(std::move(codec_ctx_)) {}

void VideoOutputStream::write_chunk(const torch::Tensor& frames) {
  for (const auto& frame : converter.convert(frames)) {
    process_frame(frame);
    frame->pts += 1;
  }
}

} // namespace torchaudio::io
