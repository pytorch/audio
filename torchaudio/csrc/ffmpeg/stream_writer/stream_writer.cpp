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

void StreamWriter::add_audio_stream(
    int sample_rate,
    int num_channels,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<EncodingConfig>& config) {
  TORCH_CHECK(!is_open, "Output is already opened. Cannot add a new stream.");
  TORCH_INTERNAL_ASSERT(
      pFormatContext->nb_streams == processes.size(),
      "The number of encode process and the number of output streams do not match.");
  processes.emplace_back(get_audio_encode_process(
      pFormatContext,
      sample_rate,
      num_channels,
      format,
      encoder,
      encoder_option,
      encoder_format,
      config));
}

void StreamWriter::add_video_stream(
    double frame_rate,
    int width,
    int height,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<std::string>& hw_accel,
    const c10::optional<EncodingConfig>& config) {
  TORCH_CHECK(!is_open, "Output is already opened. Cannot add a new stream.");
  TORCH_INTERNAL_ASSERT(
      pFormatContext->nb_streams == processes.size(),
      "The number of encode process and the number of output streams do not match.");
  processes.emplace_back(get_video_encode_process(
      pFormatContext,
      frame_rate,
      width,
      height,
      format,
      encoder,
      encoder_option,
      encoder_format,
      hw_accel,
      config));
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
  TORCH_INTERNAL_ASSERT(
      pFormatContext->nb_streams == processes.size(),
      "The number of encode process and the number of output streams do not match.");

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

void StreamWriter::write_audio_chunk(
    int i,
    const torch::Tensor& waveform,
    const c10::optional<double>& pts) {
  TORCH_CHECK(is_open, "Output is not opened. Did you call `open` method?");
  TORCH_CHECK(
      0 <= i && i < static_cast<int>(pFormatContext->nb_streams),
      "Invalid stream index. Index must be in range of [0, ",
      pFormatContext->nb_streams,
      "). Found: ",
      i);
  TORCH_CHECK(
      pFormatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO,
      "Stream ",
      i,
      " is not audio type.");
  processes[i].process(waveform, pts);
}

void StreamWriter::write_video_chunk(
    int i,
    const torch::Tensor& frames,
    const c10::optional<double>& pts) {
  TORCH_CHECK(is_open, "Output is not opened. Did you call `open` method?");
  TORCH_CHECK(
      0 <= i && i < static_cast<int>(pFormatContext->nb_streams),
      "Invalid stream index. Index must be in range of [0, ",
      pFormatContext->nb_streams,
      "). Found: ",
      i);
  TORCH_CHECK(
      pFormatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO,
      "Stream ",
      i,
      " is not video type.");
  processes[i].process(frames, pts);
}

void StreamWriter::flush() {
  TORCH_CHECK(is_open, "Output is not opened. Did you call `open` method?");
  for (auto& p : processes) {
    p.flush();
  }
}

} // namespace io
} // namespace torchaudio
