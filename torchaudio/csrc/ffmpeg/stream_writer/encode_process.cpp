#include <torchaudio/csrc/ffmpeg/stream_writer/encode_process.h>

namespace torchaudio::io {

////////////////////////////////////////////////////////////////////////////////
// EncodeProcess Logic Implementation
////////////////////////////////////////////////////////////////////////////////

EncodeProcess::EncodeProcess(
    TensorConverter&& converter,
    AVFramePtr&& frame,
    FilterGraph&& filter_graph,
    Encoder&& encoder,
    AVCodecContextPtr&& codec_ctx) noexcept
    : converter(std::move(converter)),
      src_frame(std::move(frame)),
      filter(std::move(filter_graph)),
      encoder(std::move(encoder)),
      codec_ctx(std::move(codec_ctx)) {}

void EncodeProcess::process(
    const torch::Tensor& tensor,
    const c10::optional<double>& pts) {
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

////////////////////////////////////////////////////////////////////////////////
// EncodeProcess Initialization helper functions
////////////////////////////////////////////////////////////////////////////////

namespace {

enum AVSampleFormat get_sample_fmt(const std::string& src) {
  auto fmt = av_get_sample_fmt(src.c_str());
  if (fmt != AV_SAMPLE_FMT_NONE && !av_sample_fmt_is_planar(fmt)) {
    return fmt;
  }
  TORCH_CHECK(
      false,
      "Unsupported sample fotmat (",
      src,
      ") was provided. Valid values are ",
      []() -> std::string {
        std::vector<std::string> ret;
        for (const auto& fmt :
             {AV_SAMPLE_FMT_U8,
              AV_SAMPLE_FMT_S16,
              AV_SAMPLE_FMT_S32,
              AV_SAMPLE_FMT_S64,
              AV_SAMPLE_FMT_FLT,
              AV_SAMPLE_FMT_DBL}) {
          ret.emplace_back(av_get_sample_fmt_name(fmt));
        }
        return c10::Join(", ", ret);
      }(),
      ".");
}

enum AVPixelFormat get_pix_fmt(const std::string& src) {
  AVPixelFormat fmt = av_get_pix_fmt(src.c_str());
  switch (fmt) {
    case AV_PIX_FMT_GRAY8:
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
    case AV_PIX_FMT_YUV444P:
      return fmt;
    default:;
  }
  TORCH_CHECK(
      false,
      "Unsupported pixel format (",
      src,
      ") was provided. Valid values are ",
      []() -> std::string {
        std::vector<std::string> ret;
        for (const auto& fmt :
             {AV_PIX_FMT_GRAY8,
              AV_PIX_FMT_RGB24,
              AV_PIX_FMT_BGR24,
              AV_PIX_FMT_YUV444P}) {
          ret.emplace_back(av_get_pix_fmt_name(fmt));
        }
        return c10::Join(", ", ret);
      }(),
      ".");
}

////////////////////////////////////////////////////////////////////////////////
// Codec & Codec context
////////////////////////////////////////////////////////////////////////////////
const AVCodec* get_codec(
    AVCodecID default_codec,
    const c10::optional<std::string>& encoder) {
  if (encoder) {
    const AVCodec* c = avcodec_find_encoder_by_name(encoder.value().c_str());
    TORCH_CHECK(c, "Unexpected codec: ", encoder.value());
    return c;
  }
  const AVCodec* c = avcodec_find_encoder(default_codec);
  TORCH_CHECK(
      c, "Encoder not found for codec: ", avcodec_get_name(default_codec));
  return c;
}

AVCodecContextPtr get_codec_ctx(const AVCodec* codec, int flags) {
  AVCodecContext* ctx = avcodec_alloc_context3(codec);
  TORCH_CHECK(ctx, "Failed to allocate CodecContext.");

  if (flags & AVFMT_GLOBALHEADER) {
    ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
  return AVCodecContextPtr(ctx);
}

void open_codec(
    AVCodecContext* codec_ctx,
    const c10::optional<OptionDict>& option) {
  AVDictionary* opt = get_option_dict(option);

  // Enable experimental feature if required
  // Note:
  // "vorbis" refers to FFmpeg's native encoder,
  // https://ffmpeg.org/doxygen/4.1/vorbisenc_8c.html#a8c2e524b0f125f045fef39c747561450
  // while "libvorbis" refers to the one depends on libvorbis,
  // which is not experimental
  // https://ffmpeg.org/doxygen/4.1/libvorbisenc_8c.html#a5dd5fc671e2df9c5b1f97b2ee53d4025
  // similarly, "opus" refers to FFmpeg's native encoder
  // https://ffmpeg.org/doxygen/4.1/opusenc_8c.html#a05b203d4a9a231cc1fd5a7ddeb68cebc
  // while "libopus" refers to the one depends on libopusenc
  // https://ffmpeg.org/doxygen/4.1/libopusenc_8c.html#aa1d649e48cd2ec00cfe181cf9d0f3251
  if (std::strcmp(codec_ctx->codec->name, "vorbis") == 0) {
    if (!av_dict_get(opt, "strict", nullptr, 0)) {
      TORCH_WARN_ONCE(
          "\"vorbis\" encoder is selected. Enabling '-strict experimental'. ",
          "If this is not desired, please provide \"strict\" encoder option ",
          "with desired value.");
      av_dict_set(&opt, "strict", "experimental", 0);
    }
  }
  if (std::strcmp(codec_ctx->codec->name, "opus") == 0) {
    if (!av_dict_get(opt, "strict", nullptr, 0)) {
      TORCH_WARN_ONCE(
          "\"opus\" encoder is selected. Enabling '-strict experimental'. ",
          "If this is not desired, please provide \"strict\" encoder option ",
          "with desired value.");
      av_dict_set(&opt, "strict", "experimental", 0);
    }
  }

  int ret = avcodec_open2(codec_ctx, codec_ctx->codec, &opt);
  clean_up_dict(opt);
  TORCH_CHECK(ret >= 0, "Failed to open codec: (", av_err2string(ret), ")");
}

////////////////////////////////////////////////////////////////////////////////
// Audio codec
////////////////////////////////////////////////////////////////////////////////

bool supported_sample_fmt(
    const AVSampleFormat fmt,
    const AVSampleFormat* sample_fmts) {
  if (!sample_fmts) {
    return true;
  }
  while (*sample_fmts != AV_SAMPLE_FMT_NONE) {
    if (fmt == *sample_fmts) {
      return true;
    }
    ++sample_fmts;
  }
  return false;
}

std::vector<std::string> get_supported_formats(
    const AVSampleFormat* sample_fmts) {
  std::vector<std::string> ret;
  while (*sample_fmts != AV_SAMPLE_FMT_NONE) {
    ret.emplace_back(av_get_sample_fmt_name(*sample_fmts));
    ++sample_fmts;
  }
  return ret;
}

AVSampleFormat get_enc_fmt(
    AVSampleFormat src_fmt,
    const c10::optional<std::string>& encoder_format,
    const AVCodec* codec) {
  if (encoder_format) {
    auto& enc_fmt_val = encoder_format.value();
    auto fmt = av_get_sample_fmt(enc_fmt_val.c_str());
    TORCH_CHECK(
        fmt != AV_SAMPLE_FMT_NONE, "Unknown sample format: ", enc_fmt_val);
    TORCH_CHECK(
        supported_sample_fmt(fmt, codec->sample_fmts),
        codec->name,
        " does not support ",
        encoder_format.value(),
        " format. Supported values are; ",
        c10::Join(", ", get_supported_formats(codec->sample_fmts)));
    return fmt;
  }
  if (codec->sample_fmts) {
    return codec->sample_fmts[0];
  }
  return src_fmt;
};

bool supported_sample_rate(
    const int sample_rate,
    const int* supported_samplerates) {
  if (!supported_samplerates) {
    return true;
  }
  while (*supported_samplerates) {
    if (sample_rate == *supported_samplerates) {
      return true;
    }
    ++supported_samplerates;
  }
  return false;
}

std::vector<int> get_supported_samplerates(const int* supported_samplerates) {
  std::vector<int> ret;
  if (supported_samplerates) {
    while (*supported_samplerates) {
      ret.push_back(*supported_samplerates);
      ++supported_samplerates;
    }
  }
  return ret;
}

void validate_sample_rate(int sample_rate, const AVCodec* codec) {
  TORCH_CHECK(
      supported_sample_rate(sample_rate, codec->supported_samplerates),
      codec->name,
      " does not support sample rate ",
      sample_rate,
      ". Supported values are; ",
      c10::Join(", ", get_supported_samplerates(codec->supported_samplerates)));
}

std::vector<std::string> get_supported_channels(
    const uint64_t* channel_layouts) {
  std::vector<std::string> ret;
  while (*channel_layouts) {
    ret.emplace_back(av_get_channel_name(*channel_layouts));
    ++channel_layouts;
  }
  return ret;
}

uint64_t get_channel_layout(int num_channels, const AVCodec* codec) {
  if (!codec->channel_layouts) {
    return static_cast<uint64_t>(av_get_default_channel_layout(num_channels));
  }
  for (const uint64_t* it = codec->channel_layouts; *it; ++it) {
    if (av_get_channel_layout_nb_channels(*it) == num_channels) {
      return *it;
    }
  }
  TORCH_CHECK(
      false,
      "Codec ",
      codec->name,
      " does not support a channel layout consists of ",
      num_channels,
      " channels. Supported values are: ",
      c10::Join(", ", get_supported_channels(codec->channel_layouts)));
}

void configure_audio_codec_ctx(
    AVCodecContext* codec_ctx,
    AVSampleFormat format,
    int sample_rate,
    int num_channels,
    uint64_t channel_layout,
    const c10::optional<CodecConfig>& codec_config) {
  codec_ctx->sample_fmt = format;
  codec_ctx->sample_rate = sample_rate;
  codec_ctx->time_base = av_inv_q(av_d2q(sample_rate, 1 << 24));
  codec_ctx->channels = num_channels;
  codec_ctx->channel_layout = channel_layout;

  // Set optional stuff
  if (codec_config) {
    auto& cfg = codec_config.value();
    if (cfg.bit_rate > 0) {
      codec_ctx->bit_rate = cfg.bit_rate;
    }
    if (cfg.compression_level != -1) {
      codec_ctx->compression_level = cfg.compression_level;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Video codec
////////////////////////////////////////////////////////////////////////////////

bool supported_pix_fmt(const AVPixelFormat fmt, const AVPixelFormat* pix_fmts) {
  if (!pix_fmts) {
    return true;
  }
  while (*pix_fmts != AV_PIX_FMT_NONE) {
    if (fmt == *pix_fmts) {
      return true;
    }
    ++pix_fmts;
  }
  return false;
}

std::vector<std::string> get_supported_formats(const AVPixelFormat* pix_fmts) {
  std::vector<std::string> ret;
  while (*pix_fmts != AV_PIX_FMT_NONE) {
    ret.emplace_back(av_get_pix_fmt_name(*pix_fmts));
    ++pix_fmts;
  }
  return ret;
}

AVPixelFormat get_enc_fmt(
    AVPixelFormat src_fmt,
    const c10::optional<std::string>& encoder_format,
    const AVCodec* codec) {
  if (encoder_format) {
    auto fmt = get_pix_fmt(encoder_format.value());
    TORCH_CHECK(
        supported_pix_fmt(fmt, codec->pix_fmts),
        codec->name,
        " does not support ",
        encoder_format.value(),
        " format. Supported values are; ",
        c10::Join(", ", get_supported_formats(codec->pix_fmts)));
    return fmt;
  }
  if (codec->pix_fmts) {
    return codec->pix_fmts[0];
  }
  return src_fmt;
}

bool supported_frame_rate(AVRational rate, const AVRational* rates) {
  if (!rates) {
    return true;
  }
  for (; !(rates->num == 0 && rates->den == 0); ++rates) {
    if (av_cmp_q(rate, *rates) == 0) {
      return true;
    }
  }
  return false;
}

void validate_frame_rate(AVRational rate, const AVCodec* codec) {
  TORCH_CHECK(
      supported_frame_rate(rate, codec->supported_framerates),
      codec->name,
      " does not support frame rate ",
      c10::Join("/", std::array<int, 2>{rate.num, rate.den}),
      ". Supported values are; ",
      [&]() {
        std::vector<std::string> ret;
        for (auto r = codec->supported_framerates;
             !(r->num == 0 && r->den == 0);
             ++r) {
          ret.push_back(c10::Join("/", std::array<int, 2>{r->num, r->den}));
        }
        return c10::Join(", ", ret);
      }());
}

void configure_video_codec_ctx(
    AVCodecContextPtr& ctx,
    AVPixelFormat format,
    AVRational frame_rate,
    int width,
    int height,
    const c10::optional<CodecConfig>& codec_config) {
  // TODO: Review other options and make them configurable?
  // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00147
  //  - bit_rate_tolerance
  //  - mb_decisions

  ctx->pix_fmt = format;
  ctx->width = width;
  ctx->height = height;
  ctx->time_base = av_inv_q(frame_rate);

  // Set optional stuff
  if (codec_config) {
    auto& cfg = codec_config.value();
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
      device.is_cuda(),
      "Only CUDA is supported for hardware acceleration. Found: ",
      device);

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

////////////////////////////////////////////////////////////////////////////////
// AVStream
////////////////////////////////////////////////////////////////////////////////

AVStream* get_stream(AVFormatContext* format_ctx, AVCodecContext* codec_ctx) {
  AVStream* stream = avformat_new_stream(format_ctx, nullptr);
  TORCH_CHECK(stream, "Failed to allocate stream.");

  stream->time_base = codec_ctx->time_base;
  int ret = avcodec_parameters_from_context(stream->codecpar, codec_ctx);
  TORCH_CHECK(
      ret >= 0, "Failed to copy the stream parameter: ", av_err2string(ret));
  return stream;
}

////////////////////////////////////////////////////////////////////////////////
// FilterGraph
////////////////////////////////////////////////////////////////////////////////

FilterGraph get_audio_filter_graph(
    AVSampleFormat src_fmt,
    int sample_rate,
    uint64_t channel_layout,
    AVSampleFormat enc_fmt,
    int nb_samples) {
  const std::string filter_desc = [&]() -> const std::string {
    if (src_fmt == enc_fmt) {
      if (nb_samples == 0) {
        return "anull";
      } else {
        std::stringstream ss;
        ss << "asetnsamples=n=" << nb_samples << ":p=0";
        return ss.str();
      }
    } else {
      std::stringstream ss;
      ss << "aformat=" << av_get_sample_fmt_name(enc_fmt);
      if (nb_samples > 0) {
        ss << ",asetnsamples=n=" << nb_samples << ":p=0";
      }
      return ss.str();
    }
  }();

  FilterGraph f{AVMEDIA_TYPE_AUDIO};
  f.add_audio_src(src_fmt, {1, sample_rate}, sample_rate, channel_layout);
  f.add_sink();
  f.add_process(filter_desc);
  f.create_filter();
  return f;
}

FilterGraph get_video_filter_graph(
    AVPixelFormat src_fmt,
    AVRational rate,
    int width,
    int height,
    AVPixelFormat enc_fmt,
    bool is_cuda) {
  auto desc = [&]() -> std::string {
    if (src_fmt == enc_fmt || is_cuda) {
      return "null";
    } else {
      std::stringstream ss;
      ss << "format=" << av_get_pix_fmt_name(enc_fmt);
      return ss.str();
    }
  }();

  FilterGraph f{AVMEDIA_TYPE_VIDEO};
  f.add_video_src(src_fmt, av_inv_q(rate), rate, width, height, {1, 1});
  f.add_sink();
  f.add_process(desc);
  f.create_filter();
  return f;
}

////////////////////////////////////////////////////////////////////////////////
// Source frame
////////////////////////////////////////////////////////////////////////////////

AVFramePtr get_audio_frame(
    AVSampleFormat format,
    int sample_rate,
    int num_channels,
    uint64_t channel_layout,
    int nb_samples) {
  AVFramePtr frame{};
  frame->format = format;
  frame->channel_layout = channel_layout;
  frame->sample_rate = sample_rate;
  frame->nb_samples = nb_samples ? nb_samples : 1024;
  int ret = av_frame_get_buffer(frame, 0);
  TORCH_CHECK(
      ret >= 0, "Error allocating the source audio frame:", av_err2string(ret));

  // Note: `channels` attribute is not required for encoding, but
  // TensorConverter refers to it
  frame->channels = num_channels;
  frame->pts = 0;
  return frame;
}

AVFramePtr get_video_frame(AVPixelFormat src_fmt, int width, int height) {
  AVFramePtr frame{};
  frame->format = src_fmt;
  frame->width = width;
  frame->height = height;
  int ret = av_frame_get_buffer(frame, 0);
  TORCH_CHECK(
      ret >= 0, "Error allocating a video buffer :", av_err2string(ret));

  // Note: `nb_samples` attribute is not used for video, but we set it
  // anyways so that we can make the logic of PTS increment agnostic to
  // audio and video.
  frame->nb_samples = 1;
  frame->pts = 0;
  return frame;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Finally, the extern-facing API
////////////////////////////////////////////////////////////////////////////////

EncodeProcess get_audio_encode_process(
    AVFormatContext* format_ctx,
    int src_sample_rate,
    int src_num_channels,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<CodecConfig>& codec_config) {
  // 1. Check the source format, rate and channels
  const AVSampleFormat src_fmt = get_sample_fmt(format);
  TORCH_CHECK(
      src_sample_rate > 0,
      "Sample rate must be positive. Found: ",
      src_sample_rate);
  TORCH_CHECK(
      src_num_channels > 0,
      "The number of channels must be positive. Found: ",
      src_num_channels);

  // 2. Fetch codec from default or override
  TORCH_CHECK(
      format_ctx->oformat->audio_codec != AV_CODEC_ID_NONE,
      format_ctx->oformat->name,
      " does not support audio.");
  const AVCodec* codec = get_codec(format_ctx->oformat->audio_codec, encoder);

  // 3. Check that encoding sample format, sample rate and channels
  // TODO: introduce encoder_sampel_rate option and allow to change sample rate
  const AVSampleFormat enc_fmt = get_enc_fmt(src_fmt, encoder_format, codec);
  validate_sample_rate(src_sample_rate, codec);
  uint64_t channel_layout = get_channel_layout(src_num_channels, codec);

  // 4. Initialize codec context
  AVCodecContextPtr codec_ctx =
      get_codec_ctx(codec, format_ctx->oformat->flags);
  configure_audio_codec_ctx(
      codec_ctx,
      enc_fmt,
      src_sample_rate,
      src_num_channels,
      channel_layout,
      codec_config);
  open_codec(codec_ctx, encoder_option);

  // 5. Build filter graph
  FilterGraph filter_graph = get_audio_filter_graph(
      src_fmt, src_sample_rate, channel_layout, enc_fmt, codec_ctx->frame_size);

  // 6. Instantiate source frame
  AVFramePtr src_frame = get_audio_frame(
      src_fmt,
      src_sample_rate,
      src_num_channels,
      channel_layout,
      codec_ctx->frame_size);

  // 7. Instantiate Converter
  TensorConverter converter{
      AVMEDIA_TYPE_AUDIO, src_frame, src_frame->nb_samples};

  // 8. encoder
  // Note: get_stream modifies AVFormatContext and adds new stream.
  // If anything after this throws, it will leave the StreamWriter in an
  // invalid state.
  Encoder enc{format_ctx, codec_ctx, get_stream(format_ctx, codec_ctx)};

  return EncodeProcess{
      std::move(converter),
      std::move(src_frame),
      std::move(filter_graph),
      std::move(enc),
      std::move(codec_ctx)};
}

EncodeProcess get_video_encode_process(
    AVFormatContext* format_ctx,
    double frame_rate,
    int src_width,
    int src_height,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<std::string>& hw_accel,
    const c10::optional<CodecConfig>& codec_config) {
  // 1. Checkc the source format, rate and resolution
  const AVPixelFormat src_fmt = get_pix_fmt(format);
  AVRational src_rate = av_d2q(frame_rate, 1 << 24);
  TORCH_CHECK(
      src_rate.num > 0 && src_rate.den != 0,
      "Frame rate must be positive and finite. Found: ",
      frame_rate);
  TORCH_CHECK(src_width > 0, "width must be positive. Found: ", src_width);
  TORCH_CHECK(src_height > 0, "height must be positive. Found: ", src_height);

  // 2. Fetch codec from default or override
  TORCH_CHECK(
      format_ctx->oformat->video_codec != AV_CODEC_ID_NONE,
      format_ctx->oformat->name,
      " does not support video.");
  const AVCodec* codec = get_codec(format_ctx->oformat->video_codec, encoder);

  // 3. Check that encoding format, rate
  const AVPixelFormat enc_fmt = get_enc_fmt(src_fmt, encoder_format, codec);
  validate_frame_rate(src_rate, codec);

  // 4. Initialize codec context
  AVCodecContextPtr codec_ctx =
      get_codec_ctx(codec, format_ctx->oformat->flags);
  configure_video_codec_ctx(
      codec_ctx, enc_fmt, src_rate, src_width, src_height, codec_config);
  if (hw_accel) {
#ifdef USE_CUDA
    configure_hw_accel(codec_ctx, hw_accel.value());
#else
    TORCH_CHECK(
        false,
        "torchaudio is not compiled with CUDA support. ",
        "Hardware acceleration is not available.");
#endif
  }
  open_codec(codec_ctx, encoder_option);

  // 5. Build filter graph
  FilterGraph filter_graph = get_video_filter_graph(
      src_fmt, src_rate, src_width, src_height, enc_fmt, hw_accel.has_value());

  // 6. Instantiate source frame
  AVFramePtr src_frame = [&]() {
    if (codec_ctx->hw_frames_ctx) {
      AVFramePtr frame{};
      int ret = av_hwframe_get_buffer(codec_ctx->hw_frames_ctx, frame, 0);
      TORCH_CHECK(ret >= 0, "Failed to fetch CUDA frame: ", av_err2string(ret));
      return frame;
    }
    return get_video_frame(src_fmt, src_width, src_height);
  }();

  // 7. Converter
  TensorConverter converter{AVMEDIA_TYPE_VIDEO, src_frame};

  // 8. encoder
  // Note: get_stream modifies AVFormatContext and adds new stream.
  // If anything after this throws, it will leave the StreamWriter in an
  // invalid state.
  Encoder enc{format_ctx, codec_ctx, get_stream(format_ctx, codec_ctx)};

  return EncodeProcess{
      std::move(converter),
      std::move(src_frame),
      std::move(filter_graph),
      std::move(enc),
      std::move(codec_ctx)};
}

} // namespace torchaudio::io
