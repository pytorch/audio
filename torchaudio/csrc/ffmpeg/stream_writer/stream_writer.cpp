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

StreamWriter::StreamWriter(AVFormatContext* p) : pFormatContext(p) {}

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
std::vector<std::string> get_supported_pix_fmts(const AVCodec* codec) {
  std::vector<std::string> ret;
  if (codec->pix_fmts) {
    const enum AVPixelFormat* t = codec->pix_fmts;
    while (*t != AV_PIX_FMT_NONE) {
      ret.emplace_back(av_get_pix_fmt_name(*t));
      ++t;
    }
  }
  return ret;
}

std::vector<AVRational> get_supported_frame_rates(const AVCodec* codec) {
  std::vector<AVRational> ret;
  if (codec->supported_framerates) {
    const AVRational* t = codec->supported_framerates;
    while (!(t->num == 0 && t->den == 0)) {
      ret.push_back(*t);
      ++t;
    }
  }
  return ret;
}

// used to compare frame rate / sample rate.
// not a general purpose float comparison
bool is_rate_close(double rate, AVRational rational) {
  double ref =
      static_cast<double>(rational.num) / static_cast<double>(rational.den);
  // frame rates / sample rates
  static const double threshold = 0.001;
  return fabs(rate - ref) < threshold;
}

std::vector<std::string> get_supported_sample_fmts(const AVCodec* codec) {
  std::vector<std::string> ret;
  if (codec->sample_fmts) {
    const enum AVSampleFormat* t = codec->sample_fmts;
    while (*t != AV_SAMPLE_FMT_NONE) {
      ret.emplace_back(av_get_sample_fmt_name(*t));
      ++t;
    }
  }
  return ret;
}

std::vector<int> get_supported_sample_rates(const AVCodec* codec) {
  std::vector<int> ret;
  if (codec->supported_samplerates) {
    const int* t = codec->supported_samplerates;
    while (*t) {
      ret.push_back(*t);
      ++t;
    }
  }
  return ret;
}

std::vector<uint64_t> get_supported_channel_layouts(const AVCodec* codec) {
  std::vector<uint64_t> ret;
  if (codec->channel_layouts) {
    const uint64_t* t = codec->channel_layouts;
    while (*t) {
      ret.push_back(*t);
      ++t;
    }
  }
  return ret;
}

void configure_audio_codec(
    AVCodecContextPtr& ctx,
    int64_t sample_rate,
    int64_t num_channels,
    const c10::optional<std::string>& format) {
  // TODO: Review options and make them configurable?
  // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00122
  //  - bit_rate
  //  - bit_rate_tolerance

  ctx->sample_rate = [&]() -> int {
    auto rates = get_supported_sample_rates(ctx->codec);
    if (rates.empty()) {
      return static_cast<int>(sample_rate);
    }
    for (const auto& it : rates) {
      if (it == sample_rate) {
        return static_cast<int>(sample_rate);
      }
    }
    TORCH_CHECK(
        false,
        ctx->codec->name,
        " does not support sample rate ",
        sample_rate,
        ". Supported sample rates are: ",
        c10::Join(", ", rates));
  }();
  ctx->time_base = av_inv_q(av_d2q(sample_rate, 1 << 24));
  ctx->sample_fmt = [&]() {
    // Use default
    if (!format) {
      TORCH_CHECK(
          ctx->codec->sample_fmts,
          ctx->codec->name,
          " does not have default sample format. Please specify one.");
      return ctx->codec->sample_fmts[0];
    }
    // Use the given one.
    auto fmt = format.value();
    auto ret = av_get_sample_fmt(fmt.c_str());
    auto fmts = get_supported_sample_fmts(ctx->codec);
    if (fmts.empty()) {
      TORCH_CHECK(
          ret != AV_SAMPLE_FMT_NONE, "Unrecognized format: ", fmt, ". ");
      return ret;
    }
    TORCH_CHECK(
        std::count(fmts.begin(), fmts.end(), fmt),
        "Unsupported sample format: ",
        fmt,
        ". Supported values are ",
        c10::Join(", ", fmts));
    return ret;
  }();

  // validate and set channels
  ctx->channels = static_cast<int>(num_channels);
  auto layout = av_get_default_channel_layout(ctx->channels);
  auto layouts = get_supported_channel_layouts(ctx->codec);
  if (!layouts.empty()) {
    if (!std::count(layouts.begin(), layouts.end(), layout)) {
      std::vector<std::string> tmp;
      for (const auto& it : layouts) {
        tmp.push_back(std::to_string(av_get_channel_layout_nb_channels(it)));
      }
      TORCH_CHECK(
          false,
          "Unsupported channels: ",
          num_channels,
          ". Supported channels are: ",
          c10::Join(", ", tmp));
    }
  }
  ctx->channel_layout = static_cast<uint64_t>(layout);
}

void configure_video_codec(
    AVCodecContextPtr& ctx,
    double frame_rate,
    int64_t width,
    int64_t height,
    const c10::optional<std::string>& format) {
  // TODO: Review other options and make them configurable?
  // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00147
  //  - bit_rate
  //  - bit_rate_tolerance
  //  - gop_size
  //  - max_b_frames
  //  - mb_decisions

  ctx->width = static_cast<int>(width);
  ctx->height = static_cast<int>(height);
  ctx->time_base = [&]() {
    AVRational ret = av_inv_q(av_d2q(frame_rate, 1 << 24));
    auto rates = get_supported_frame_rates(ctx->codec);
    // Codec does not have constraint on frame rate
    if (rates.empty()) {
      return ret;
    }
    // Codec has list of supported frame rate.
    for (const auto& t : rates) {
      if (is_rate_close(frame_rate, t)) {
        return ret;
      }
    }
    // Given one is not supported.
    std::vector<std::string> tmp;
    for (const auto& t : rates) {
      tmp.emplace_back(
          t.den == 1 ? std::to_string(t.num)
                     : std::to_string(t.num) + "/" + std::to_string(t.den));
    }
    TORCH_CHECK(
        false,
        "Unsupported frame rate: ",
        frame_rate,
        ". Supported values are ",
        c10::Join(", ", tmp));
  }();
  ctx->pix_fmt = [&]() {
    // Use default
    if (!format) {
      TORCH_CHECK(
          ctx->codec->pix_fmts,
          ctx->codec->name,
          " does not have defaut pixel format. Please specify one.");
      return ctx->codec->pix_fmts[0];
    }
    // Use the given one,
    auto fmt = format.value();
    auto ret = av_get_pix_fmt(fmt.c_str());
    auto fmts = get_supported_pix_fmts(ctx->codec);
    if (fmts.empty()) {
      TORCH_CHECK(ret != AV_PIX_FMT_NONE, "Unrecognized format: ", fmt, ". ");
      return ret;
    }
    if (!std::count(fmts.begin(), fmts.end(), fmt)) {
      TORCH_CHECK(
          false,
          "Unsupported pixel format: ",
          fmt,
          ". Supported values are ",
          c10::Join(", ", fmts));
    }
    return ret;
  }();
}

void open_codec(
    AVCodecContextPtr& codec_ctx,
    const c10::optional<OptionDict>& option) {
  AVDictionary* opt = get_option_dict(option);
  int ret = avcodec_open2(codec_ctx, codec_ctx->codec, &opt);
  clean_up_dict(opt);
  TORCH_CHECK(ret >= 0, "Failed to open codec: (", av_err2string(ret), ")");
}

AVFramePtr get_audio_frame(
    enum AVSampleFormat fmt,
    AVCodecContextPtr& codec_ctx,
    int frame_size) {
  AVFramePtr frame{};
  frame->format = fmt;
  frame->channel_layout = codec_ctx->channel_layout;
  frame->sample_rate = codec_ctx->sample_rate;
  frame->nb_samples = frame_size;
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

AVFramePtr get_hw_video_frame(AVCodecContextPtr& codec_ctx) {
  AVFramePtr frame{};
  int ret = av_hwframe_get_buffer(codec_ctx->hw_frames_ctx, frame, 0);
  TORCH_CHECK(ret >= 0, "Failed to fetch CUDA frame: ", av_err2string(ret));
  return frame;
}

AVFramePtr get_video_frame(
    enum AVPixelFormat fmt,
    AVCodecContextPtr& codec_ctx) {
  AVFramePtr frame{};
  frame->format = fmt;
  frame->width = codec_ctx->width;
  frame->height = codec_ctx->height;

  int ret = av_frame_get_buffer(frame, 0);
  TORCH_CHECK(
      ret >= 0, "Error allocating a video buffer (", av_err2string(ret), ").");
  return frame;
}

AVCodecContextPtr get_codec_ctx(
    enum AVMediaType type,
    AVFORMAT_CONST AVOutputFormat* oformat,
    const c10::optional<std::string>& encoder) {
  enum AVCodecID default_codec = [&]() {
    switch (type) {
      case AVMEDIA_TYPE_AUDIO:
        return oformat->audio_codec;
      case AVMEDIA_TYPE_VIDEO:
        return oformat->video_codec;
      default:
        TORCH_CHECK(
            false, "Unsupported media type: ", av_get_media_type_string(type));
    }
  }();

  TORCH_CHECK(
      default_codec != AV_CODEC_ID_NONE,
      "Format \"",
      oformat->name,
      "\" does not support ",
      av_get_media_type_string(type),
      ".");

  const AVCodec* codec = [&]() {
    if (encoder) {
      const AVCodec* c = avcodec_find_encoder_by_name(encoder.value().c_str());
      TORCH_CHECK(c, "Unexpected codec: ", encoder.value());
      return c;
    }
    const AVCodec* c = avcodec_find_encoder(default_codec);
    TORCH_CHECK(
        c, "Encoder not found for codec: ", avcodec_get_name(default_codec));
    return c;
  }();

  AVCodecContext* ctx = avcodec_alloc_context3(codec);
  TORCH_CHECK(ctx, "Failed to allocate CodecContext.");

  if (oformat->flags & AVFMT_GLOBALHEADER) {
    ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
  return AVCodecContextPtr(ctx);
}

enum AVSampleFormat _get_src_sample_fmt(const std::string& src) {
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

enum AVPixelFormat _get_src_pixel_fmt(const std::string& src) {
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

std::unique_ptr<FilterGraph> _get_audio_filter(
    enum AVSampleFormat fmt,
    AVCodecContextPtr& ctx) {
  std::stringstream desc;
  desc << "aformat=" << av_get_sample_fmt_name(ctx->sample_fmt);

  auto p = std::make_unique<FilterGraph>(AVMEDIA_TYPE_AUDIO);
  p->add_audio_src(fmt, ctx->time_base, ctx->sample_rate, ctx->channel_layout);
  p->add_sink();
  p->add_process(desc.str());
  p->create_filter();
  return p;
}

std::unique_ptr<FilterGraph> _get_video_filter(
    enum AVPixelFormat fmt,
    AVCodecContextPtr& ctx) {
  std::stringstream desc;
  desc << "format=" << av_get_pix_fmt_name(ctx->pix_fmt);

  auto p = std::make_unique<FilterGraph>(AVMEDIA_TYPE_VIDEO);
  p->add_video_src(
      fmt, ctx->time_base, ctx->width, ctx->height, ctx->sample_aspect_ratio);
  p->add_sink();
  p->add_process(desc.str());
  p->create_filter();
  return p;
}

} // namespace

void StreamWriter::add_audio_stream(
    int64_t sample_rate,
    int64_t num_channels,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format) {
  enum AVSampleFormat src_fmt = _get_src_sample_fmt(format);

  AVCodecContextPtr ctx =
      get_codec_ctx(AVMEDIA_TYPE_AUDIO, pFormatContext->oformat, encoder);
  configure_audio_codec(ctx, sample_rate, num_channels, encoder_format);
  open_codec(ctx, encoder_option);
  AVStream* stream = add_stream(ctx);

  std::unique_ptr<FilterGraph> filter = src_fmt == ctx->sample_fmt
      ? std::unique_ptr<FilterGraph>(nullptr)
      : _get_audio_filter(src_fmt, ctx);
  static const int default_capacity = 10000;
  int frame_capacity = ctx->frame_size ? ctx->frame_size : default_capacity;
  AVFramePtr src_frame = get_audio_frame(src_fmt, ctx, frame_capacity);
  streams.emplace_back(std::make_unique<AudioOutputStream>(
      pFormatContext,
      stream,
      std::move(ctx),
      std::move(filter),
      std::move(src_frame),
      frame_capacity));
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
  const torch::Device device = [&]() {
    if (!hw_accel) {
      return torch::Device{c10::DeviceType::CPU};
    }
#ifdef USE_CUDA
    torch::Device d{hw_accel.value()};
    TORCH_CHECK(
        d.type() == c10::DeviceType::CUDA,
        "Only CUDA is supported for hardware acceleration. Found:",
        device.str());
    return d;
#else
    TORCH_CHECK(
        false,
        "torchaudio is not compiled with CUDA support. Hardware acceleration is not available.");
#endif
  }();
  enum AVPixelFormat src_fmt = _get_src_pixel_fmt(format);

  AVCodecContextPtr ctx =
      get_codec_ctx(AVMEDIA_TYPE_VIDEO, pFormatContext->oformat, encoder);
  configure_video_codec(ctx, frame_rate, width, height, encoder_format);

  AVBufferRefPtr hw_device_ctx{};
  AVBufferRefPtr hw_frame_ctx{};
#ifdef USE_CUDA
  if (device.type() == c10::DeviceType::CUDA) {
    AVBufferRef* device_ctx = nullptr;
    int ret = av_hwdevice_ctx_create(
        &device_ctx,
        AV_HWDEVICE_TYPE_CUDA,
        std::to_string(device.index()).c_str(),
        nullptr,
        0);
    TORCH_CHECK(
        ret >= 0, "Failed to create CUDA device context: ", av_err2string(ret));
    hw_device_ctx.reset(device_ctx);

    AVBufferRef* frames_ref = av_hwframe_ctx_alloc(device_ctx);
    TORCH_CHECK(frames_ref, "Failed to create CUDA frame context.");
    hw_frame_ctx.reset(frames_ref);

    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)(frames_ref->data);
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = ctx->pix_fmt;
    frames_ctx->width = ctx->width;
    frames_ctx->height = ctx->height;
    frames_ctx->initial_pool_size = 20;
    ctx->sw_pix_fmt = ctx->pix_fmt;
    ctx->pix_fmt = AV_PIX_FMT_CUDA;

    ret = av_hwframe_ctx_init(frames_ref);
    TORCH_CHECK(
        ret >= 0,
        "Failed to initialize CUDA frame context: ",
        av_err2string(ret));

    ctx->hw_frames_ctx = av_buffer_ref(frames_ref);
    TORCH_CHECK(
        ctx->hw_frames_ctx,
        "Failed to attach CUDA frames to encoding context: ",
        av_err2string(ret));
  }
#endif

  open_codec(ctx, encoder_option);
  AVStream* stream = add_stream(ctx);

  std::unique_ptr<FilterGraph> filter = [&]() {
    if (src_fmt != ctx->pix_fmt && device.type() == c10::DeviceType::CPU) {
      return _get_video_filter(src_fmt, ctx);
    }
    return std::unique_ptr<FilterGraph>(nullptr);
  }();

  AVFramePtr src_frame = [&]() {
    if (device.type() == c10::DeviceType::CUDA) {
      return get_hw_video_frame(ctx);
    }
    return get_video_frame(src_fmt, ctx);
  }();
  streams.emplace_back(std::make_unique<VideoOutputStream>(
      pFormatContext,
      stream,
      std::move(ctx),
      std::move(filter),
      std::move(src_frame),
      std::move(hw_device_ctx),
      std::move(hw_frame_ctx)));
}

AVStream* StreamWriter::add_stream(AVCodecContextPtr& codec_ctx) {
  AVStream* stream = avformat_new_stream(pFormatContext, nullptr);
  TORCH_CHECK(stream, "Failed to allocate stream.");

  stream->time_base = codec_ctx->time_base;
  int ret = avcodec_parameters_from_context(stream->codecpar, codec_ctx);
  TORCH_CHECK(
      ret >= 0,
      "Failed to copy the stream parameter. (",
      av_err2string(ret),
      ")");
  return stream;
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
}

void StreamWriter::validate_stream(int i, enum AVMediaType type) {
  TORCH_CHECK(
      0 <= i && i < static_cast<int>(streams.size()),
      "Invalid stream index. Index must be in range of [0, ",
      streams.size(),
      "). Found: ",
      i);

  TORCH_CHECK(
      streams[i]->stream->codecpar->codec_type == type,
      "Stream ",
      i,
      " is not ",
      av_get_media_type_string(type));
}

void StreamWriter::write_audio_chunk(int i, const torch::Tensor& waveform) {
  validate_stream(i, AVMEDIA_TYPE_AUDIO);
  streams[i]->write_chunk(waveform);
}

void StreamWriter::write_video_chunk(int i, const torch::Tensor& frames) {
  validate_stream(i, AVMEDIA_TYPE_VIDEO);
  streams[i]->write_chunk(frames);
}

void StreamWriter::flush() {
  for (auto& os : streams) {
    os->flush();
  }
}

} // namespace io
} // namespace torchaudio
