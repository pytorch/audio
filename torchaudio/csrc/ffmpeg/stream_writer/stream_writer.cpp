#include <torchaudio/csrc/ffmpeg/stream_writer/stream_writer.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace torchaudio {
namespace io {
namespace {

AVFormatContext* get_output_format_context(
    const std::string& dst,
    const c10::optional<std::string>& format,
    AVIOContext* io_ctx) {
  if (io_ctx) {
    TORCH_CHECK(
        format,
        "`format` must be provided when the input is file-like object.");
  }

  AVFormatContext* p = nullptr;
  int ret = avformat_alloc_output_context2(
      &p, nullptr, format ? format.value().c_str() : nullptr, dst.c_str());
  TORCH_CHECK(
      ret >= 0,
      "Failed to open output \"",
      dst,
      "\" (",
      av_err2string(ret),
      ").");

  if (io_ctx) {
    p->pb = io_ctx;
    p->flags |= AVFMT_FLAG_CUSTOM_IO;
  }

  return p;
}
} // namespace

StreamWriter::StreamWriter(AVFormatContext* p) : pFormatContext(p) {
  C10_LOG_API_USAGE_ONCE("torchaudio.io.StreamWriter");
}

StreamWriter::StreamWriter(
    AVIOContext* io_ctx,
    const c10::optional<std::string>& format)
    : StreamWriter(
          get_output_format_context("Custom Output Context", format, io_ctx)) {}

StreamWriter::StreamWriter(
    const std::string& dst,
    const c10::optional<std::string>& format)
    : StreamWriter(get_output_format_context(dst, format, nullptr)) {}

namespace {

enum AVSampleFormat get_src_sample_fmt(const std::string& src) {
  auto fmt = av_get_sample_fmt(src.c_str());
  TORCH_CHECK(fmt != AV_SAMPLE_FMT_NONE, "Unknown sample format: ", src);
  TORCH_CHECK(
      !av_sample_fmt_is_planar(fmt),
      "Unexpected sample fotmat value. Valid values are ",
      av_get_sample_fmt_name(AV_SAMPLE_FMT_U8),
      ", ",
      av_get_sample_fmt_name(AV_SAMPLE_FMT_S16),
      ", ",
      av_get_sample_fmt_name(AV_SAMPLE_FMT_S32),
      ", ",
      av_get_sample_fmt_name(AV_SAMPLE_FMT_S64),
      ", ",
      av_get_sample_fmt_name(AV_SAMPLE_FMT_FLT),
      ", ",
      av_get_sample_fmt_name(AV_SAMPLE_FMT_DBL),
      ". ",
      "Found: ",
      src);
  return fmt;
}

enum AVPixelFormat get_src_pixel_fmt(const std::string& src) {
  auto fmt = av_get_pix_fmt(src.c_str());
  switch (fmt) {
    case AV_PIX_FMT_GRAY8:
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
    case AV_PIX_FMT_YUV444P:
      return fmt;
    case AV_PIX_FMT_NONE:
      TORCH_CHECK(false, "Unknown pixel format: ", src);
    default:
      TORCH_CHECK(false, "Unsupported pixel format: ", src);
  }
}

} // namespace

void StreamWriter::add_audio_stream(
    int64_t sample_rate,
    int64_t num_channels,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format) {
  processes.emplace_back(
      pFormatContext,
      sample_rate,
      num_channels,
      get_src_sample_fmt(format),
      encoder,
      encoder_option,
      encoder_format);
}

void StreamWriter::add_video_stream(
    double frame_rate,
    int64_t width,
    int64_t height,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<std::string>& hw_accel) {
  processes.emplace_back(
      pFormatContext,
      frame_rate,
      width,
      height,
      get_src_pixel_fmt(format),
      encoder,
      encoder_option,
      encoder_format,
      hw_accel);
}

void StreamWriter::set_metadata(const OptionDict& metadata) {
  av_dict_free(&pFormatContext->metadata);
  for (auto const& [key, value] : metadata) {
    av_dict_set(&pFormatContext->metadata, key.c_str(), value.c_str(), 0);
  }
}

void StreamWriter::dump_format(int64_t i) {
  av_dump_format(pFormatContext, (int)i, pFormatContext->url, 1);
}

void StreamWriter::open(const c10::optional<OptionDict>& option) {
  int ret = 0;

  // Open the file if it was not provided by client code (i.e. when not
  // file-like object)
  AVFORMAT_CONST AVOutputFormat* fmt = pFormatContext->oformat;
  AVDictionary* opt = get_option_dict(option);
  if (!(fmt->flags & AVFMT_NOFILE) &&
      !(pFormatContext->flags & AVFMT_FLAG_CUSTOM_IO)) {
    ret = avio_open2(
        &pFormatContext->pb,
        pFormatContext->url,
        AVIO_FLAG_WRITE,
        nullptr,
        &opt);
    if (ret < 0) {
      av_dict_free(&opt);
      TORCH_CHECK(
          false,
          "Failed to open dst: ",
          pFormatContext->url,
          " (",
          av_err2string(ret),
          ")");
    }
  }

  ret = avformat_write_header(pFormatContext, &opt);
  clean_up_dict(opt);
  TORCH_CHECK(
      ret >= 0,
      "Failed to write header: ",
      pFormatContext->url,
      " (",
      av_err2string(ret),
      ")");
  is_open = true;
}

void StreamWriter::close() {
  int ret = av_write_trailer(pFormatContext);
  if (ret < 0) {
    LOG(WARNING) << "Failed to write trailer. (" << av_err2string(ret) << ").";
  }

  // Close the file if it was not provided by client code (i.e. when not
  // file-like object)
  AVFORMAT_CONST AVOutputFormat* fmt = pFormatContext->oformat;
  if (!(fmt->flags & AVFMT_NOFILE) &&
      !(pFormatContext->flags & AVFMT_FLAG_CUSTOM_IO)) {
    // avio_closep can be only applied to AVIOContext opened by avio_open
    avio_closep(&(pFormatContext->pb));
  }
  is_open = false;
}

void StreamWriter::write_audio_chunk(int i, const torch::Tensor& waveform) {
  TORCH_CHECK(is_open, "Output is not opened. Did you call `open` method?");
  TORCH_CHECK(
      0 <= i && i < static_cast<int>(processes.size()),
      "Invalid stream index. Index must be in range of [0, ",
      processes.size(),
      "). Found: ",
      i);
  processes[i].process(AVMEDIA_TYPE_AUDIO, waveform);
}

void StreamWriter::write_video_chunk(int i, const torch::Tensor& frames) {
  TORCH_CHECK(is_open, "Output is not opened. Did you call `open` method?");
  TORCH_CHECK(
      0 <= i && i < static_cast<int>(processes.size()),
      "Invalid stream index. Index must be in range of [0, ",
      processes.size(),
      "). Found: ",
      i);
  processes[i].process(AVMEDIA_TYPE_VIDEO, frames);
}

void StreamWriter::flush() {
  TORCH_CHECK(is_open, "Output is not opened. Did you call `open` method?");
  for (auto& p : processes) {
    p.flush();
  }
}

} // namespace io
} // namespace torchaudio
