#include <torchaudio/csrc/ffmpeg/stream_writer/encode_process.h>

namespace torchaudio::io {

namespace {

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
    const c10::optional<std::string>& format,
    const c10::optional<EncodingConfig>& config) {
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

  // Set optional stuff
  if (config) {
    auto& cfg = config.value();
    if (cfg.bit_rate > 0) {
      ctx->bit_rate = cfg.bit_rate;
    }
    if (cfg.compression_level != -1) {
      ctx->compression_level = cfg.compression_level;
    }
  }
}

void open_codec(
    AVCodecContextPtr& codec_ctx,
    const c10::optional<OptionDict>& option) {
  AVDictionary* opt = get_option_dict(option);
  int ret = avcodec_open2(codec_ctx, codec_ctx->codec, &opt);
  clean_up_dict(opt);
  TORCH_CHECK(ret >= 0, "Failed to open codec: (", av_err2string(ret), ")");
}

AVCodecContextPtr get_audio_codec(
    AVFORMAT_CONST AVOutputFormat* oformat,
    int64_t sample_rate,
    int64_t num_channels,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<EncodingConfig>& config) {
  AVCodecContextPtr ctx = get_codec_ctx(AVMEDIA_TYPE_AUDIO, oformat, encoder);
  configure_audio_codec(ctx, sample_rate, num_channels, encoder_format, config);
  open_codec(ctx, encoder_option);
  return ctx;
}

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
    int sample_rate,
    int num_channels,
    AVCodecContext* codec_ctx,
    int default_frame_size = 10000) {
  AVFramePtr frame{};
  frame->pts = 0;
  frame->format = src_fmt;
  // Note: `channels` attribute is not required for encoding, but
  // TensorConverter refers to it
  frame->channels = num_channels;
  frame->channel_layout = codec_ctx->channel_layout;
  frame->sample_rate = sample_rate;
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

void configure_video_codec(
    AVCodecContextPtr& ctx,
    double frame_rate,
    int64_t width,
    int64_t height,
    const c10::optional<std::string>& format,
    const c10::optional<EncodingConfig>& config) {
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

  // Set optional stuff
  if (config) {
    auto& cfg = config.value();
    if (cfg.bit_rate > 0) {
      ctx->bit_rate = cfg.bit_rate;
    }
    if (cfg.compression_level != -1) {
      ctx->compression_level = cfg.compression_level;
    }
    if (cfg.gop_size != -1) {
      ctx->gop_size = cfg.gop_size;
    }
    if (cfg.max_b_frames != -1) {
      ctx->max_b_frames = cfg.max_b_frames;
    }
  }
}

void configure_hw_accel(AVCodecContext* ctx, const std::string& hw_accel) {
  torch::Device device{hw_accel};
  TORCH_CHECK(
      device.type() == c10::DeviceType::CUDA,
      "Only CUDA is supported for hardware acceleration. Found: ",
      device.str());

  // NOTES:
  // 1. Examples like
  // https://ffmpeg.org/doxygen/4.1/hw_decode_8c-example.html#a9 wraps the HW
  // device context and the HW frames context with av_buffer_ref. This
  // increments the reference counting and the resource won't be automatically
  // dallocated at the time AVCodecContex is destructed. (We will need to
  // decrement once ourselves), so we do not do it. When adding support to share
  // context objects, this needs to be reviewed.
  //
  // 2. When encoding, it is technically not necessary to attach HW device
  // context to AVCodecContext. But this way, it will be deallocated
  // automatically at the time AVCodecContext is freed, so we do that.

  int ret = av_hwdevice_ctx_create(
      &ctx->hw_device_ctx,
      AV_HWDEVICE_TYPE_CUDA,
      std::to_string(device.index()).c_str(),
      nullptr,
      0);
  TORCH_CHECK(
      ret >= 0, "Failed to create CUDA device context: ", av_err2string(ret));
  assert(ctx->hw_device_ctx);

  ctx->sw_pix_fmt = ctx->pix_fmt;
  ctx->pix_fmt = AV_PIX_FMT_CUDA;

  ctx->hw_frames_ctx = av_hwframe_ctx_alloc(ctx->hw_device_ctx);
  TORCH_CHECK(ctx->hw_frames_ctx, "Failed to create CUDA frame context.");

  auto frames_ctx = (AVHWFramesContext*)(ctx->hw_frames_ctx->data);
  frames_ctx->format = ctx->pix_fmt;
  frames_ctx->sw_format = ctx->sw_pix_fmt;
  frames_ctx->width = ctx->width;
  frames_ctx->height = ctx->height;
  frames_ctx->initial_pool_size = 5;

  ret = av_hwframe_ctx_init(ctx->hw_frames_ctx);
  TORCH_CHECK(
      ret >= 0,
      "Failed to initialize CUDA frame context: ",
      av_err2string(ret));
}

AVCodecContextPtr get_video_codec(
    AVFORMAT_CONST AVOutputFormat* oformat,
    double frame_rate,
    int64_t width,
    int64_t height,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<std::string>& hw_accel,
    const c10::optional<EncodingConfig>& config) {
  AVCodecContextPtr ctx = get_codec_ctx(AVMEDIA_TYPE_VIDEO, oformat, encoder);
  configure_video_codec(ctx, frame_rate, width, height, encoder_format, config);

  if (hw_accel) {
#ifdef USE_CUDA
    configure_hw_accel(ctx, hw_accel.value());
#else
    TORCH_CHECK(
        false,
        "torchaudio is not compiled with CUDA support. ",
        "Hardware acceleration is not available.");
#endif
  }

  open_codec(ctx, encoder_option);
  return ctx;
}

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
      codec_ctx->framerate,
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
  // Note: `nb_samples` attribute is not used for video, but we set it
  // anyways so that we can make the logic of PTS increment agnostic to
  // audio and video.
  frame->nb_samples = 1;
  frame->pts = 0;
  return frame;
}

} // namespace

EncodeProcess::EncodeProcess(
    AVFormatContext* format_ctx,
    int sample_rate,
    int num_channels,
    const enum AVSampleFormat format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<EncodingConfig>& config)
    : codec_ctx(get_audio_codec(
          format_ctx->oformat,
          sample_rate,
          num_channels,
          encoder,
          encoder_option,
          encoder_format,
          config)),
      encoder(format_ctx, codec_ctx),
      filter(get_audio_filter(format, codec_ctx)),
      src_frame(get_audio_frame(format, sample_rate, num_channels, codec_ctx)),
      converter(AVMEDIA_TYPE_AUDIO, src_frame, src_frame->nb_samples) {}

EncodeProcess::EncodeProcess(
    AVFormatContext* format_ctx,
    double frame_rate,
    int width,
    int height,
    const enum AVPixelFormat format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<std::string>& hw_accel,
    const c10::optional<EncodingConfig>& config)
    : codec_ctx(get_video_codec(
          format_ctx->oformat,
          frame_rate,
          width,
          height,
          encoder,
          encoder_option,
          encoder_format,
          hw_accel,
          config)),
      encoder(format_ctx, codec_ctx),
      filter(get_video_filter(format, codec_ctx)),
      src_frame(get_video_frame(format, codec_ctx)),
      converter(AVMEDIA_TYPE_VIDEO, src_frame) {}

void EncodeProcess::process(
    AVMediaType type,
    const torch::Tensor& tensor,
    const c10::optional<double>& pts) {
  TORCH_CHECK(
      codec_ctx->codec_type == type,
      "Attempted to write ",
      av_get_media_type_string(type),
      " to ",
      av_get_media_type_string(codec_ctx->codec_type),
      " stream.");
  if (pts) {
    AVRational tb = codec_ctx->time_base;
    auto val = static_cast<int64_t>(std::round(pts.value() * tb.den / tb.num));
    if (src_frame->pts > val) {
      TORCH_WARN_ONCE(
          "The provided PTS value is smaller than the next expected value.");
    }
    src_frame->pts = val;
  }
  for (const auto& frame : converter.convert(tensor)) {
    process_frame(frame);
    frame->pts += frame->nb_samples;
  }
}

void EncodeProcess::process_frame(AVFrame* src) {
  int ret = filter.add_frame(src);
  while (ret >= 0) {
    ret = filter.get_frame(dst_frame);
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

void EncodeProcess::flush() {
  process_frame(nullptr);
}

} // namespace torchaudio::io
