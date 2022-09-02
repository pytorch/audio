#include <torchaudio/csrc/ffmpeg/stream_writer/stream_writer.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace torchaudio {
namespace ffmpeg {
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

} // namespace

StreamWriter::StreamWriter(AVFormatOutputContextPtr&& p)
    : pFormatContext(std::move(p)), streams(), pkt() {}

namespace {
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
  ctx->time_base = AVRational{1, static_cast<int>(sample_rate)};
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
    AVRational ret = AVRational{1, static_cast<int>(frame_rate)};
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
  TORCH_CHECK(
      ret >= 0, "Failed to open audio codec: (", av_err2string(ret), ")");
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
  AVFramePtr dst_frame = filter
      ? AVFramePtr{}
      : get_audio_frame(ctx->sample_fmt, ctx, frame_capacity);
  streams.emplace_back(OutputStream{
      stream,
      std::move(ctx),
      std::move(filter),
      std::move(src_frame),
      std::move(dst_frame),
      0,
      frame_capacity,
      AVBufferRefPtr{},
      AVBufferRefPtr{}});
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

  // CUDA: require src_frame
  // CPU: require dst_frame when filter is enabled
  AVFramePtr src_frame = [&]() {
    if (device.type() == c10::DeviceType::CUDA) {
      return get_hw_video_frame(ctx);
    }
    return get_video_frame(src_fmt, ctx);
  }();
  AVFramePtr dst_frame =
      filter ? get_video_frame(ctx->pix_fmt, ctx) : AVFramePtr{};
  streams.emplace_back(OutputStream{
      stream,
      std::move(ctx),
      std::move(filter),
      std::move(src_frame),
      std::move(dst_frame),
      0,
      -1,
      std::move(hw_device_ctx),
      std::move(hw_frame_ctx)});
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
  for (const auto& it : metadata) {
    av_dict_set(
        &pFormatContext->metadata, it.key().c_str(), it.value().c_str(), 0);
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
      streams[i].stream->codecpar->codec_type == type,
      "Stream ",
      i,
      " is not ",
      av_get_media_type_string(type));
}

void StreamWriter::process_frame(
    AVFrame* src_frame,
    std::unique_ptr<FilterGraph>& filter,
    AVFrame* dst_frame,
    AVCodecContextPtr& c,
    AVStream* st) {
  int ret = filter->add_frame(src_frame);
  while (ret >= 0) {
    ret = filter->get_frame(dst_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    }
    if (ret >= 0) {
      encode_frame(dst_frame, c, st);
    }
    av_frame_unref(dst_frame);
  }
}

void StreamWriter::encode_frame(
    AVFrame* frame,
    AVCodecContextPtr& c,
    AVStream* st) {
  int ret = avcodec_send_frame(c, frame);
  TORCH_CHECK(ret >= 0, "Failed to encode frame (", av_err2string(ret), ").");
  while (ret >= 0) {
    ret = avcodec_receive_packet(c, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    } else {
      TORCH_CHECK(
          ret >= 0,
          "Failed to fetch encoded packet (",
          av_err2string(ret),
          ").");
    }
    av_packet_rescale_ts(pkt, c->time_base, st->time_base);
    pkt->stream_index = st->index;

    ret = av_interleaved_write_frame(pFormatContext, pkt);
    TORCH_CHECK(ret >= 0, "Failed to write packet (", av_err2string(ret), ").");
  }
}

namespace {
void validate_audio_input(
    enum AVSampleFormat fmt,
    AVCodecContext* ctx,
    const torch::Tensor& t) {
  auto dtype = t.dtype().toScalarType();
  switch (fmt) {
    case AV_SAMPLE_FMT_U8:
      TORCH_CHECK(
          dtype == c10::ScalarType::Byte, "Expected Tensor of uint8 type.");
      break;
    case AV_SAMPLE_FMT_S16:
      TORCH_CHECK(
          dtype == c10::ScalarType::Short, "Expected Tensor of int16 type.");
      break;
    case AV_SAMPLE_FMT_S32:
      TORCH_CHECK(
          dtype == c10::ScalarType::Int, "Expected Tensor of int32 type.");
      break;
    case AV_SAMPLE_FMT_S64:
      TORCH_CHECK(
          dtype == c10::ScalarType::Long, "Expected Tensor of int64 type.");
      break;
    case AV_SAMPLE_FMT_FLT:
      TORCH_CHECK(
          dtype == c10::ScalarType::Float, "Expected Tensor of float32 type.");
      break;
    case AV_SAMPLE_FMT_DBL:
      TORCH_CHECK(
          dtype == c10::ScalarType::Double, "Expected Tensor of float64 type.");
      break;
    default:
      TORCH_CHECK(
          false,
          "Internal error: Audio encoding stream is not properly configured.");
  }
  TORCH_CHECK(t.device().is_cpu(), "Input tensor has to be on CPU.");
  TORCH_CHECK(t.dim() == 2, "Input Tensor has to be 2D.");
  const auto num_channels = t.size(1);
  TORCH_CHECK(
      num_channels == ctx->channels,
      "Expected waveform with ",
      ctx->channels,
      " channels. Found ",
      num_channels);
}

void validate_video_input(
    enum AVPixelFormat fmt,
    AVCodecContext* ctx,
    const torch::Tensor& t) {
  auto dtype = t.dtype().toScalarType();
  TORCH_CHECK(dtype == c10::ScalarType::Byte, "Expected Tensor of uint8 type.");
  TORCH_CHECK(t.dim() == 4, "Input Tensor has to be 4D.");

  // Note: the number of color components is not same as the number of planes.
  // For example, YUV420P has only two planes. U and V are in the second plane.
  int num_color_components = av_pix_fmt_desc_get(fmt)->nb_components;

  const auto channels = t.size(1);
  const auto height = t.size(2);
  const auto width = t.size(3);
  TORCH_CHECK(
      channels == num_color_components && height == ctx->height &&
          width == ctx->width,
      "Expected tensor with shape (N, ",
      num_color_components,
      ", ",
      ctx->height,
      ", ",
      ctx->width,
      ") (NCHW format). Found ",
      t.sizes());
}
} // namespace

void StreamWriter::write_audio_chunk(int i, const torch::Tensor& waveform) {
  validate_stream(i, AVMEDIA_TYPE_AUDIO);
  OutputStream& os = streams[i];

  validate_audio_input(
      static_cast<AVSampleFormat>(os.src_frame->format),
      os.codec_ctx,
      waveform);

  const auto num_frames = waveform.size(0);
  int64_t num_unit_frames = os.frame_capacity;

  AVRational time_base{1, os.codec_ctx->sample_rate};

  using namespace torch::indexing;
  AT_DISPATCH_ALL_TYPES(waveform.scalar_type(), "write_audio_frames", [&] {
    for (int64_t i = 0; i < num_frames; i += num_unit_frames) {
      auto chunk = waveform.index({Slice(i, i + num_unit_frames), Slice()});
      auto num_valid_frames = chunk.size(0);
      auto byte_size = chunk.numel() * chunk.element_size();
      chunk = chunk.reshape({-1}).contiguous();

      // TODO: make writable
      // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00334
      TORCH_CHECK(
          av_frame_is_writable(os.src_frame),
          "Internal Error: frame is not writable.");

      memcpy(
          os.src_frame->data[0],
          static_cast<void*>(chunk.data_ptr<scalar_t>()),
          byte_size);
      os.src_frame->pts =
          av_rescale_q(os.num_frames, time_base, os.codec_ctx->time_base);
      os.src_frame->nb_samples = num_valid_frames;
      os.num_frames += num_valid_frames;

      if (os.filter) {
        process_frame(
            os.src_frame, os.filter, os.dst_frame, os.codec_ctx, os.stream);
      } else {
        encode_frame(os.src_frame, os.codec_ctx, os.stream);
      }
    }
  });
}

void StreamWriter::write_video_chunk(int i, const torch::Tensor& frames) {
  validate_stream(i, AVMEDIA_TYPE_VIDEO);
  OutputStream& os = streams[i];
  enum AVPixelFormat fmt = static_cast<AVPixelFormat>(os.src_frame->format);

#ifdef USE_CUDA
  if (fmt == AV_PIX_FMT_CUDA) {
    TORCH_CHECK(frames.device().is_cuda(), "Input tensor has to be on CUDA.");
    enum AVPixelFormat sw_fmt = os.codec_ctx->sw_pix_fmt;
    validate_video_input(sw_fmt, os.codec_ctx, frames);
    switch (sw_fmt) {
      case AV_PIX_FMT_RGB0:
      case AV_PIX_FMT_BGR0:
        write_interlaced_video_cuda(os, frames, true);
        return;
      case AV_PIX_FMT_GBRP:
      case AV_PIX_FMT_GBRP16LE:
      case AV_PIX_FMT_YUV444P:
      case AV_PIX_FMT_YUV444P16LE:
        write_planar_video_cuda(os, frames, av_pix_fmt_count_planes(sw_fmt));
        return;
      default:
        TORCH_CHECK(
            false,
            "Unexpected pixel format for CUDA: ",
            av_get_pix_fmt_name(sw_fmt));
    }
  }
#endif

  TORCH_CHECK(frames.device().is_cpu(), "Input tensor has to be on CPU.");
  validate_video_input(fmt, os.codec_ctx, frames);
  switch (fmt) {
    case AV_PIX_FMT_GRAY8:
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
      write_interlaced_video(os, frames);
      return;
    case AV_PIX_FMT_YUV444P:
      write_planar_video(os, frames, av_pix_fmt_count_planes(fmt));
      return;
    default:
      TORCH_CHECK(false, "Unexpected pixel format: ", av_get_pix_fmt_name(fmt));
  }
}

#ifdef USE_CUDA
void StreamWriter::write_interlaced_video_cuda(
    OutputStream& os,
    const torch::Tensor& frames,
    bool pad_extra) {
  const auto num_frames = frames.size(0);
  const auto num_channels = frames.size(1);
  const auto height = frames.size(2);
  const auto width = frames.size(3);
  const auto num_channels_buffer = num_channels + (pad_extra ? 1 : 0);

  using namespace torch::indexing;
  torch::Tensor buffer =
      torch::empty({height, width, num_channels_buffer}, frames.options());
  size_t spitch = width * num_channels_buffer;
  for (int i = 0; i < num_frames; ++i) {
    // Slice frame as HWC
    auto chunk = frames.index({i}).permute({1, 2, 0});
    buffer.index_put_({"...", Slice(0, num_channels)}, chunk);

    if (cudaSuccess !=
        cudaMemcpy2D(
            (void*)(os.src_frame->data[0]),
            os.src_frame->linesize[0],
            (const void*)(buffer.data_ptr<uint8_t>()),
            spitch,
            spitch,
            height,
            cudaMemcpyDeviceToDevice)) {
      TORCH_CHECK(false, "Failed to copy pixel data from CUDA tensor.");
    }
    os.src_frame->pts = os.num_frames;
    os.num_frames += 1;
    encode_frame(os.src_frame, os.codec_ctx, os.stream);
  }
}

void StreamWriter::write_planar_video_cuda(
    OutputStream& os,
    const torch::Tensor& frames,
    int num_planes) {
  const auto num_frames = frames.size(0);
  const auto height = frames.size(2);
  const auto width = frames.size(3);

  using namespace torch::indexing;
  torch::Tensor buffer = torch::empty({height, width}, frames.options());
  for (int i = 0; i < num_frames; ++i) {
    for (int j = 0; j < num_planes; ++j) {
      buffer.index_put_({"..."}, frames.index({i, j}));
      if (cudaSuccess !=
          cudaMemcpy2D(
              (void*)(os.src_frame->data[j]),
              os.src_frame->linesize[j],
              (const void*)(buffer.data_ptr<uint8_t>()),
              width,
              width,
              height,
              cudaMemcpyDeviceToDevice)) {
        TORCH_CHECK(false, "Failed to copy pixel data from CUDA tensor.");
      }
    }
    os.src_frame->pts = os.num_frames;
    os.num_frames += 1;
    encode_frame(os.src_frame, os.codec_ctx, os.stream);
  }
}
#endif

// Interlaced video
// Each frame is composed of one plane, and color components for each pixel are
// collocated.
// The memory layout is 1D linear, interpretated as following.
//
//    |<----- linesize[0] ----->|
//      0   1 ...   W
// 0: RGB RGB ... RGB PAD ... PAD
// 1: RGB RGB ... RGB PAD ... PAD
//            ...
// H: RGB RGB ... RGB PAD ... PAD
void StreamWriter::write_interlaced_video(
    OutputStream& os,
    const torch::Tensor& frames) {
  const auto num_frames = frames.size(0);
  const auto num_channels = frames.size(1);
  const auto height = frames.size(2);
  const auto width = frames.size(3);

  using namespace torch::indexing;
  size_t stride = width * num_channels;
  for (int i = 0; i < num_frames; ++i) {
    // TODO: writable
    // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00472
    TORCH_CHECK(
        av_frame_is_writable(os.src_frame),
        "Internal Error: frame is not writable.");

    // CHW -> HWC
    auto chunk =
        frames.index({i}).permute({1, 2, 0}).reshape({-1}).contiguous();

    uint8_t* src = chunk.data_ptr<uint8_t>();
    uint8_t* dst = os.src_frame->data[0];
    for (int h = 0; h < height; ++h) {
      std::memcpy(dst, src, stride);
      src += width * num_channels;
      dst += os.src_frame->linesize[0];
    }
    os.src_frame->pts = os.num_frames;
    os.num_frames += 1;

    if (os.filter) {
      process_frame(
          os.src_frame, os.filter, os.dst_frame, os.codec_ctx, os.stream);
    } else {
      encode_frame(os.src_frame, os.codec_ctx, os.stream);
    }
  }
}

// Planar video
// Each frame is composed of multiple planes.
// One plane can contain one of more color components.
// (but at the moment only accept formats without subsampled color components)
//
// The memory layout is interpreted as follow
//
//    |<----- linesize[0] ----->|
//       0   1 ...  W1
//  0:   Y   Y ...   Y PAD ... PAD
//  1:   Y   Y ...   Y PAD ... PAD
//             ...
// H1:   Y   Y ...   Y PAD ... PAD
//
//    |<--- linesize[1] ---->|
//       0 ...  W2
//  0:  UV ...  UV PAD ... PAD
//  1:  UV ...  UV PAD ... PAD
//         ...
// H2:  UV ...  UV PAD ... PAD
//
void StreamWriter::write_planar_video(
    OutputStream& os,
    const torch::Tensor& frames,
    int num_planes) {
  const auto num_frames = frames.size(0);
  const auto height = frames.size(2);
  const auto width = frames.size(3);

  using namespace torch::indexing;
  for (int i = 0; i < num_frames; ++i) {
    // TODO: writable
    // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00472
    TORCH_CHECK(
        av_frame_is_writable(os.src_frame),
        "Internal Error: frame is not writable.");

    for (int j = 0; j < num_planes; ++j) {
      auto chunk = frames.index({i, j}).contiguous();

      uint8_t* src = chunk.data_ptr<uint8_t>();
      uint8_t* dst = os.src_frame->data[j];
      for (int h = 0; h < height; ++h) {
        memcpy(dst, src, width);
        src += width;
        dst += os.src_frame->linesize[j];
      }
    }
    os.src_frame->pts = os.num_frames;
    os.num_frames += 1;

    if (os.filter) {
      process_frame(
          os.src_frame, os.filter, os.dst_frame, os.codec_ctx, os.stream);
    } else {
      encode_frame(os.src_frame, os.codec_ctx, os.stream);
    }
  }
}

// TODO: probably better to flush output streams in interweaving manner.
void StreamWriter::flush() {
  for (auto& os : streams) {
    flush_stream(os);
  }
}

void StreamWriter::flush_stream(OutputStream& os) {
  if (os.filter) {
    process_frame(nullptr, os.filter, os.dst_frame, os.codec_ctx, os.stream);
  }
  encode_frame(nullptr, os.codec_ctx, os.stream);
}
} // namespace ffmpeg
} // namespace torchaudio
