#include <torchaudio/csrc/ffmpeg/hw_context.h>
#include <torchaudio/csrc/ffmpeg/stream_writer/encode_process.h>
#include <cmath>

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
    const double& pts_val = pts.value();
    TORCH_CHECK(
        std::isfinite(pts_val) && pts_val >= 0.0,
        "The value of PTS must be positive and finite. Found: ",
        pts_val)
    AVRational tb = codec_ctx->time_base;
    auto val = static_cast<int64_t>(std::round(pts_val * tb.den / tb.num));
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

enum AVSampleFormat get_src_sample_fmt(const std::string& src) {
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

enum AVPixelFormat get_src_pix_fmt(const std::string& src) {
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

std::string get_supported_formats(const AVSampleFormat* sample_fmts) {
  std::vector<std::string> ret;
  while (*sample_fmts != AV_SAMPLE_FMT_NONE) {
    ret.emplace_back(av_get_sample_fmt_name(*sample_fmts));
    ++sample_fmts;
  }
  return c10::Join(", ", ret);
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
        get_supported_formats(codec->sample_fmts));
    return fmt;
  }
  if (codec->sample_fmts) {
    return codec->sample_fmts[0];
  }
  return src_fmt;
};

bool supported_sample_rate(const int sample_rate, const AVCodec* codec) {
  if (!codec->supported_samplerates) {
    return true;
  }
  const int* it = codec->supported_samplerates;
  while (*it) {
    if (sample_rate == *it) {
      return true;
    }
    ++it;
  }
  return false;
}

std::string get_supported_samplerates(const int* supported_samplerates) {
  std::vector<int> ret;
  if (supported_samplerates) {
    while (*supported_samplerates) {
      ret.push_back(*supported_samplerates);
      ++supported_samplerates;
    }
  }
  return c10::Join(", ", ret);
}

int get_enc_sr(
    int src_sample_rate,
    const c10::optional<int>& encoder_sample_rate,
    const AVCodec* codec) {
  // G.722 only supports 16000 Hz, but it does not list the sample rate in
  // supported_samplerates so we hard code it here.
  if (codec->id == AV_CODEC_ID_ADPCM_G722) {
    if (encoder_sample_rate) {
      auto val = encoder_sample_rate.value();
      TORCH_CHECK(
          val == 16'000,
          codec->name,
          " does not support sample rate ",
          val,
          ". Supported values are; 16000.");
    }
    return 16'000;
  }
  if (encoder_sample_rate) {
    const int& encoder_sr = encoder_sample_rate.value();
    TORCH_CHECK(
        encoder_sr > 0,
        "Encoder sample rate must be positive. Found: ",
        encoder_sr);
    TORCH_CHECK(
        supported_sample_rate(encoder_sr, codec),
        codec->name,
        " does not support sample rate ",
        encoder_sr,
        ". Supported values are; ",
        get_supported_samplerates(codec->supported_samplerates));
    return encoder_sr;
  }
  if (codec->supported_samplerates &&
      !supported_sample_rate(src_sample_rate, codec)) {
    return codec->supported_samplerates[0];
  }
  return src_sample_rate;
}

std::string get_supported_channels(const uint64_t* channel_layouts) {
  std::vector<std::string> names;
  while (*channel_layouts) {
    std::stringstream ss;
    ss << av_get_channel_layout_nb_channels(*channel_layouts);
    ss << " (" << av_get_channel_name(*channel_layouts) << ")";
    names.emplace_back(ss.str());
    ++channel_layouts;
  }
  return c10::Join(", ", names);
}

uint64_t get_channel_layout(
    const uint64_t src_ch_layout,
    const c10::optional<int> enc_num_channels,
    const AVCodec* codec) {
  // If the override is presented, and if it is supported by codec, we use it.
  if (enc_num_channels) {
    const int& val = enc_num_channels.value();
    TORCH_CHECK(
        val > 0, "The number of channels must be greater than 0. Found: ", val);
    if (!codec->channel_layouts) {
      return static_cast<uint64_t>(av_get_default_channel_layout(val));
    }
    for (const uint64_t* it = codec->channel_layouts; *it; ++it) {
      if (av_get_channel_layout_nb_channels(*it) == val) {
        return *it;
      }
    }
    TORCH_CHECK(
        false,
        "Codec ",
        codec->name,
        " does not support a channel layout consists of ",
        val,
        " channels. Supported values are: ",
        get_supported_channels(codec->channel_layouts));
  }
  // If the codec does not have restriction on channel layout, we reuse the
  // source channel layout
  if (!codec->channel_layouts) {
    return src_ch_layout;
  }
  // If the codec has restriction, and source layout is supported, we reuse the
  // source channel layout
  for (const uint64_t* it = codec->channel_layouts; *it; ++it) {
    if (*it == src_ch_layout) {
      return src_ch_layout;
    }
  }
  // Use the default layout of the codec.
  return codec->channel_layouts[0];
}

void configure_audio_codec_ctx(
    AVCodecContext* codec_ctx,
    AVSampleFormat format,
    int sample_rate,
    uint64_t channel_layout,
    const c10::optional<CodecConfig>& codec_config) {
  codec_ctx->sample_fmt = format;
  codec_ctx->sample_rate = sample_rate;
  codec_ctx->time_base = av_inv_q(av_d2q(sample_rate, 1 << 24));
  codec_ctx->channels = av_get_channel_layout_nb_channels(channel_layout);
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
    if (cfg.qscale) {
      codec_ctx->flags |= AV_CODEC_FLAG_QSCALE;
      codec_ctx->global_quality = FF_QP2LAMBDA * cfg.qscale.value();
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

std::string get_supported_formats(const AVPixelFormat* pix_fmts) {
  std::vector<std::string> ret;
  while (*pix_fmts != AV_PIX_FMT_NONE) {
    ret.emplace_back(av_get_pix_fmt_name(*pix_fmts));
    ++pix_fmts;
  }
  return c10::Join(", ", ret);
}

AVPixelFormat get_enc_fmt(
    AVPixelFormat src_fmt,
    const c10::optional<std::string>& encoder_format,
    const AVCodec* codec) {
  if (encoder_format) {
    const auto& val = encoder_format.value();
    auto fmt = av_get_pix_fmt(val.c_str());
    TORCH_CHECK(
        supported_pix_fmt(fmt, codec->pix_fmts),
        codec->name,
        " does not support ",
        val,
        " format. Supported values are; ",
        get_supported_formats(codec->pix_fmts));
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

AVRational get_enc_rate(
    AVRational src_rate,
    const c10::optional<double>& encoder_sample_rate,
    const AVCodec* codec) {
  if (encoder_sample_rate) {
    const double& enc_rate = encoder_sample_rate.value();
    TORCH_CHECK(
        std::isfinite(enc_rate) && enc_rate > 0,
        "Encoder sample rate must be positive and fininte. Found: ",
        enc_rate);
    AVRational rate = av_d2q(enc_rate, 1 << 24);
    TORCH_CHECK(
        supported_frame_rate(rate, codec->supported_framerates),
        codec->name,
        " does not support frame rate: ",
        enc_rate,
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
    return rate;
  }
  if (codec->supported_framerates &&
      !supported_frame_rate(src_rate, codec->supported_framerates)) {
    return codec->supported_framerates[0];
  }
  return src_rate;
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
    if (cfg.qscale) {
      ctx->flags |= AV_CODEC_FLAG_QSCALE;
      ctx->global_quality = FF_QP2LAMBDA * cfg.qscale.value();
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

  ctx->hw_device_ctx = av_buffer_ref(get_cuda_context(device.index()));
  TORCH_INTERNAL_ASSERT(
      ctx->hw_device_ctx, "Failed to reference HW device context.");

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

  int ret = av_hwframe_ctx_init(ctx->hw_frames_ctx);
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
    int src_sample_rate,
    uint64_t src_ch_layout,
    const c10::optional<std::string>& filter_desc,
    AVSampleFormat enc_fmt,
    int enc_sample_rate,
    uint64_t enc_ch_layout,
    int nb_samples) {
  const auto desc = [&]() -> const std::string {
    std::vector<std::string> parts;
    if (filter_desc) {
      parts.push_back(filter_desc.value());
    }
    if (filter_desc || src_fmt != enc_fmt ||
        src_sample_rate != enc_sample_rate || src_ch_layout != enc_ch_layout) {
      std::stringstream ss;
      ss << "aformat=sample_fmts=" << av_get_sample_fmt_name(enc_fmt)
         << ":sample_rates=" << enc_sample_rate << ":channel_layouts=0x"
         << std::hex << enc_ch_layout;
      parts.push_back(ss.str());
    }
    if (nb_samples > 0) {
      std::stringstream ss;
      ss << "asetnsamples=n=" << nb_samples << ":p=0";
      parts.push_back(ss.str());
    }
    if (parts.size()) {
      return c10::Join(",", parts);
    }
    return "anull";
  }();

  FilterGraph f;
  f.add_audio_src(
      src_fmt, {1, src_sample_rate}, src_sample_rate, src_ch_layout);
  f.add_audio_sink();
  f.add_process(desc);
  f.create_filter();
  return f;
}

FilterGraph get_video_filter_graph(
    AVPixelFormat src_fmt,
    AVRational src_rate,
    int src_width,
    int src_height,
    const c10::optional<std::string>& filter_desc,
    AVPixelFormat enc_fmt,
    AVRational enc_rate,
    int enc_width,
    int enc_height,
    bool is_cuda) {
  const auto desc = [&]() -> const std::string {
    if (is_cuda) {
      return filter_desc.value_or("null");
    }
    std::vector<std::string> parts;
    if (filter_desc) {
      parts.push_back(filter_desc.value());
    }
    if (filter_desc || (src_width != enc_width || src_height != enc_height)) {
      std::stringstream ss;
      ss << "scale=" << enc_width << ":" << enc_height;
      parts.emplace_back(ss.str());
    }
    if (filter_desc || src_fmt != enc_fmt) {
      std::stringstream ss;
      ss << "format=" << av_get_pix_fmt_name(enc_fmt);
      parts.emplace_back(ss.str());
    }
    if (filter_desc ||
        (src_rate.num != enc_rate.num || src_rate.den != enc_rate.den)) {
      std::stringstream ss;
      ss << "fps=" << enc_rate.num << "/" << enc_rate.den;
      parts.emplace_back(ss.str());
    }
    if (parts.size()) {
      return c10::Join(",", parts);
    }
    return "null";
  }();

  FilterGraph f;
  f.add_video_src(
      is_cuda ? AV_PIX_FMT_CUDA : src_fmt,
      av_inv_q(src_rate),
      src_rate,
      src_width,
      src_height,
      {1, 1});
  f.add_video_sink();
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
  AVFramePtr frame{alloc_avframe()};
  frame->format = format;
  frame->channel_layout = channel_layout;
  frame->sample_rate = sample_rate;
  frame->nb_samples = nb_samples;
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
  AVFramePtr frame{alloc_avframe()};
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
    const c10::optional<int>& encoder_sample_rate,
    const c10::optional<int>& encoder_num_channels,
    const c10::optional<CodecConfig>& codec_config,
    const c10::optional<std::string>& filter_desc,
    bool disable_converter) {
  // 1. Check the source format, rate and channels
  TORCH_CHECK(
      src_sample_rate > 0,
      "Sample rate must be positive. Found: ",
      src_sample_rate);
  TORCH_CHECK(
      src_num_channels > 0,
      "The number of channels must be positive. Found: ",
      src_num_channels);
  // Note that disable_converter = true indicates that the caller is looking to
  // directly supply frames and bypass tensor conversion. Therefore, in this
  // case, restrictions on the format to support tensor inputs do not apply, and
  // so we directly get the format via FFmpeg.
  const AVSampleFormat src_fmt = (disable_converter)
      ? av_get_sample_fmt(format.c_str())
      : get_src_sample_fmt(format);
  const auto src_ch_layout =
      static_cast<uint64_t>(av_get_default_channel_layout(src_num_channels));

  // 2. Fetch codec from default or override
  TORCH_CHECK(
      format_ctx->oformat->audio_codec != AV_CODEC_ID_NONE,
      format_ctx->oformat->name,
      " does not support audio.");
  const AVCodec* codec = get_codec(format_ctx->oformat->audio_codec, encoder);

  // 3. Check that encoding sample format, sample rate and channels
  const AVSampleFormat enc_fmt = get_enc_fmt(src_fmt, encoder_format, codec);
  const int enc_sr = get_enc_sr(src_sample_rate, encoder_sample_rate, codec);
  const uint64_t enc_ch_layout = [&]() -> uint64_t {
    if (std::strcmp(codec->name, "vorbis") == 0) {
      // Special case for vorbis.
      // It only supports 2 channels, but it is not listed in channel_layouts
      // attributes.
      // https://github.com/FFmpeg/FFmpeg/blob/0684e58886881a998f1a7b510d73600ff1df2b90/libavcodec/vorbisenc.c#L1277
      // This is the case for at least until FFmpeg 6.0, so it will be
      // like this for a while.
      return static_cast<uint64_t>(av_get_default_channel_layout(2));
    }
    return get_channel_layout(src_ch_layout, encoder_num_channels, codec);
  }();

  // 4. Initialize codec context
  AVCodecContextPtr codec_ctx =
      get_codec_ctx(codec, format_ctx->oformat->flags);
  configure_audio_codec_ctx(
      codec_ctx, enc_fmt, enc_sr, enc_ch_layout, codec_config);
  open_codec(codec_ctx, encoder_option);

  // 5. Build filter graph
  FilterGraph filter_graph = get_audio_filter_graph(
      src_fmt,
      src_sample_rate,
      src_ch_layout,
      filter_desc,
      enc_fmt,
      enc_sr,
      enc_ch_layout,
      codec_ctx->frame_size);

  // 6. Instantiate source frame
  AVFramePtr src_frame = get_audio_frame(
      src_fmt,
      src_sample_rate,
      src_num_channels,
      src_ch_layout,
      codec_ctx->frame_size > 0 ? codec_ctx->frame_size : 256);

  // 7. Instantiate Converter
  TensorConverter converter{
      (disable_converter) ? AVMEDIA_TYPE_UNKNOWN : AVMEDIA_TYPE_AUDIO,
      src_frame,
      src_frame->nb_samples};

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

namespace {

bool ends_with(std::string_view str, std::string_view suffix) {
  return str.size() >= suffix.size() &&
      0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

} // namespace

EncodeProcess get_video_encode_process(
    AVFormatContext* format_ctx,
    double frame_rate,
    int src_width,
    int src_height,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<double>& encoder_frame_rate,
    const c10::optional<int>& encoder_width,
    const c10::optional<int>& encoder_height,
    const c10::optional<std::string>& hw_accel,
    const c10::optional<CodecConfig>& codec_config,
    const c10::optional<std::string>& filter_desc,
    bool disable_converter) {
  // 1. Checkc the source format, rate and resolution
  TORCH_CHECK(
      std::isfinite(frame_rate) && frame_rate > 0,
      "Frame rate must be positive and finite. Found: ",
      frame_rate);
  TORCH_CHECK(src_width > 0, "width must be positive. Found: ", src_width);
  TORCH_CHECK(src_height > 0, "height must be positive. Found: ", src_height);
  // Note that disable_converter = true indicates that the caller is looking to
  // directly supply frames and bypass tensor conversion. Therefore, in this
  // case, restrictions on the format to support tensor inputs do not apply, and
  // so we directly get the format via FFmpeg.
  const AVPixelFormat src_fmt = (disable_converter)
      ? av_get_pix_fmt(format.c_str())
      : get_src_pix_fmt(format);
  const AVRational src_rate = av_d2q(frame_rate, 1 << 24);

  // 2. Fetch codec from default or override
  TORCH_CHECK(
      format_ctx->oformat->video_codec != AV_CODEC_ID_NONE,
      format_ctx->oformat->name,
      " does not support video.");
  const AVCodec* codec = get_codec(format_ctx->oformat->video_codec, encoder);

  // 3. Check that encoding format, rate
  const AVPixelFormat enc_fmt = get_enc_fmt(src_fmt, encoder_format, codec);
  const AVRational enc_rate = get_enc_rate(src_rate, encoder_frame_rate, codec);
  const int enc_width = [&]() -> int {
    if (!encoder_width) {
      return src_width;
    }
    const int& val = encoder_width.value();
    TORCH_CHECK(val > 0, "Encoder width must be positive. Found: ", val);
    return val;
  }();
  const int enc_height = [&]() -> int {
    if (!encoder_height) {
      return src_height;
    }
    const int& val = encoder_height.value();
    TORCH_CHECK(val > 0, "Encoder height must be positive. Found: ", val);
    return val;
  }();

  // 4. Initialize codec context
  AVCodecContextPtr codec_ctx =
      get_codec_ctx(codec, format_ctx->oformat->flags);
  configure_video_codec_ctx(
      codec_ctx, enc_fmt, enc_rate, enc_width, enc_height, codec_config);
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

  if (ends_with(codec_ctx->codec->name, "_nvenc")) {
    C10_LOG_API_USAGE_ONCE("torchaudio.io.StreamReaderCUDA");
  }

  // 5. Build filter graph
  FilterGraph filter_graph = get_video_filter_graph(
      src_fmt,
      src_rate,
      src_width,
      src_height,
      filter_desc,
      enc_fmt,
      enc_rate,
      enc_width,
      enc_height,
      hw_accel.has_value());

  // 6. Instantiate source frame
  AVFramePtr src_frame = [&]() {
    if (codec_ctx->hw_frames_ctx) {
      AVFramePtr frame{alloc_avframe()};
      int ret = av_hwframe_get_buffer(codec_ctx->hw_frames_ctx, frame, 0);
      TORCH_CHECK(ret >= 0, "Failed to fetch CUDA frame: ", av_err2string(ret));
      frame->nb_samples = 1;
      frame->pts = 0;
      return frame;
    }
    return get_video_frame(src_fmt, src_width, src_height);
  }();

  // 7. Converter
  TensorConverter converter{
      (disable_converter) ? AVMEDIA_TYPE_UNKNOWN : AVMEDIA_TYPE_VIDEO,
      src_frame};

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
