#include <torchaudio/csrc/ffmpeg/hw_context.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/stream_processor.h>
#include <stdexcept>
#include <string_view>

namespace torchaudio::io {

namespace {
AVCodecContextPtr alloc_codec_context(
    enum AVCodecID codec_id,
    const c10::optional<std::string>& decoder_name) {
  const AVCodec* codec = [&]() {
    if (decoder_name) {
      const AVCodec* c =
          avcodec_find_decoder_by_name(decoder_name.value().c_str());
      TORCH_CHECK(c, "Unsupported codec: ", decoder_name.value());
      return c;
    } else {
      const AVCodec* c = avcodec_find_decoder(codec_id);
      TORCH_CHECK(c, "Unsupported codec: ", avcodec_get_name(codec_id));
      return c;
    }
  }();

  AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
  TORCH_CHECK(codec_ctx, "Failed to allocate CodecContext.");
  return AVCodecContextPtr(codec_ctx);
}

const AVCodecHWConfig* get_cuda_config(const AVCodec* codec) {
  for (int i = 0;; ++i) {
    const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
    if (!config) {
      break;
    }
    if (config->device_type == AV_HWDEVICE_TYPE_CUDA &&
        config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
      return config;
    }
  }
  TORCH_CHECK(
      false,
      "CUDA device was requested, but the codec \"",
      codec->name,
      "\" is not supported.");
}

enum AVPixelFormat get_hw_format(
    AVCodecContext* codec_ctx,
    const enum AVPixelFormat* pix_fmts) {
  const AVCodecHWConfig* cfg = static_cast<AVCodecHWConfig*>(codec_ctx->opaque);
  for (const enum AVPixelFormat* p = pix_fmts; *p != -1; p++) {
    if (*p == cfg->pix_fmt) {
      // Note
      // The HW decode example uses generic approach
      // https://ffmpeg.org/doxygen/4.1/hw__decode_8c_source.html#l00063
      // But this approach finalizes the codec configuration when the first
      // frame comes in.
      // We need to inspect the codec configuration right after the codec is
      // opened.
      // So we add short cut for known patterns.
      // yuv420p (h264) -> nv12
      // yuv420p10le (hevc/h265) -> p010le
      switch (codec_ctx->pix_fmt) {
        case AV_PIX_FMT_YUV420P: {
          codec_ctx->pix_fmt = AV_PIX_FMT_CUDA;
          codec_ctx->sw_pix_fmt = AV_PIX_FMT_NV12;
          break;
        }
        case AV_PIX_FMT_YUV420P10LE: {
          codec_ctx->pix_fmt = AV_PIX_FMT_CUDA;
          codec_ctx->sw_pix_fmt = AV_PIX_FMT_P010LE;
          break;
        }
        default:;
      }
      return *p;
    }
  }
  TORCH_WARN("Failed to get HW surface format.");
  return AV_PIX_FMT_NONE;
}

AVBufferRef* get_hw_frames_ctx(AVCodecContext* codec_ctx) {
  AVBufferRef* p = av_hwframe_ctx_alloc(codec_ctx->hw_device_ctx);
  TORCH_CHECK(
      p,
      "Failed to allocate CUDA frame context from device context at ",
      codec_ctx->hw_device_ctx);
  auto frames_ctx = (AVHWFramesContext*)(p->data);
  frames_ctx->format = codec_ctx->pix_fmt;
  frames_ctx->sw_format = codec_ctx->sw_pix_fmt;
  frames_ctx->width = codec_ctx->width;
  frames_ctx->height = codec_ctx->height;
  frames_ctx->initial_pool_size = 5;
  int ret = av_hwframe_ctx_init(p);
  if (ret >= 0) {
    return p;
  }
  av_buffer_unref(&p);
  TORCH_CHECK(
      false, "Failed to initialize CUDA frame context: ", av_err2string(ret));
}

void configure_codec_context(
    AVCodecContext* codec_ctx,
    const AVCodecParameters* params,
    const torch::Device& device) {
  int ret = avcodec_parameters_to_context(codec_ctx, params);
  TORCH_CHECK(
      ret >= 0, "Failed to set CodecContext parameter: ", av_err2string(ret));

  if (device.type() == c10::DeviceType::CUDA) {
#ifndef USE_CUDA
    TORCH_CHECK(false, "torchaudio is not compiled with CUDA support.");
#else
    const AVCodecHWConfig* cfg = get_cuda_config(codec_ctx->codec);
    // https://www.ffmpeg.org/doxygen/trunk/hw__decode_8c_source.html#l00221
    // 1. Set HW config to opaue pointer.
    codec_ctx->opaque = static_cast<void*>(const_cast<AVCodecHWConfig*>(cfg));
    // 2. Set pCodecContext->get_format call back function which
    // will retrieve the HW pixel format from opaque pointer.
    codec_ctx->get_format = get_hw_format;
    codec_ctx->hw_device_ctx = av_buffer_ref(get_cuda_context(device.index()));
    TORCH_INTERNAL_ASSERT(
        codec_ctx->hw_device_ctx, "Failed to reference HW device context.");
#endif
  }
}

void open_codec(
    AVCodecContext* codec_ctx,
    const c10::optional<OptionDict>& decoder_option) {
  AVDictionary* opts = get_option_dict(decoder_option);

  // Default to single thread execution.
  if (!av_dict_get(opts, "threads", nullptr, 0)) {
    av_dict_set(&opts, "threads", "1", 0);
  }

  if (!codec_ctx->channel_layout) {
    codec_ctx->channel_layout =
        av_get_default_channel_layout(codec_ctx->channels);
  }

  int ret = avcodec_open2(codec_ctx, codec_ctx->codec, &opts);
  clean_up_dict(opts);
  TORCH_CHECK(
      ret >= 0, "Failed to initialize CodecContext: ", av_err2string(ret));
}

bool ends_with(std::string_view str, std::string_view suffix) {
  return str.size() >= suffix.size() &&
      0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

AVCodecContextPtr get_codec_ctx(
    const AVCodecParameters* params,
    const c10::optional<std::string>& decoder_name,
    const c10::optional<OptionDict>& decoder_option,
    const torch::Device& device) {
  AVCodecContextPtr codec_ctx =
      alloc_codec_context(params->codec_id, decoder_name);
  configure_codec_context(codec_ctx, params, device);
  open_codec(codec_ctx, decoder_option);
  if (codec_ctx->hw_device_ctx) {
    codec_ctx->hw_frames_ctx = get_hw_frames_ctx(codec_ctx);
  }
  if (ends_with(codec_ctx->codec->name, "_cuvid")) {
    C10_LOG_API_USAGE_ONCE("torchaudio.io.StreamReaderCUDA");
  }
  return codec_ctx;
}

} // namespace

using KeyType = StreamProcessor::KeyType;

StreamProcessor::StreamProcessor(const AVRational& time_base)
    : stream_time_base(time_base) {}

////////////////////////////////////////////////////////////////////////////////
// Configurations
////////////////////////////////////////////////////////////////////////////////
KeyType StreamProcessor::add_stream(
    int frames_per_chunk,
    int num_chunks,
    AVRational frame_rate,
    const std::string& filter_description,
    const torch::Device& device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      is_decoder_set(), "Decoder hasn't been set.");
  // If device is provided, then check that codec_ctx has hw_device_ctx set.
  // In case, defining an output stream with HW accel on an input stream that
  // has decoder set without HW accel, it will cause seg fault.
  // i.e.
  // The following should be rejected here.
  // reader = StreamReader(...)
  // reader.add_video_stream(..., decoder="h264_cuvid")
  // reader.add_video_stream(..., decoder="h264_cuvid", hw_accel="cuda")
  // TODO:
  // One idea to work around this is to always define HW device context, and
  // if HW acceleration is not required, insert `hwdownload` filter.
  // This way it will be possible to handle both cases at the same time.
  switch (device.type()) {
    case torch::kCPU:
      TORCH_CHECK(
          !codec_ctx->hw_device_ctx,
          "Decoding without Hardware acceleration is requested, however, "
          "the decoder has been already defined with a HW acceleration. "
          "Decoding a stream with and without HW acceleration simultaneously "
          "is not supported.");
      break;
    case torch::kCUDA:
      TORCH_CHECK(
          codec_ctx->hw_device_ctx,
          "CUDA Hardware acceleration is requested, however, the decoder has "
          "been already defined without a HW acceleration. "
          "Decoding a stream with and without HW acceleration simultaneously "
          "is not supported.");
      break;
    default:;
  }

  switch (codec_ctx->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
      post_processes.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(current_key),
          std::forward_as_tuple(get_audio_process(
              stream_time_base,
              codec_ctx,
              filter_description,
              frames_per_chunk,
              num_chunks)));
      return current_key++;
    case AVMEDIA_TYPE_VIDEO:
      post_processes.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(current_key),
          std::forward_as_tuple(get_video_process(
              stream_time_base,
              frame_rate,
              codec_ctx,
              filter_description,
              frames_per_chunk,
              num_chunks,
              device)));
      return current_key++;
    default:
      TORCH_CHECK(false, "Only Audio and Video are supported");
  }
}

void StreamProcessor::remove_stream(KeyType key) {
  post_processes.erase(key);
}

void StreamProcessor::set_discard_timestamp(int64_t timestamp) {
  TORCH_CHECK(timestamp >= 0, "timestamp must be non-negative.");
  discard_before_pts =
      av_rescale_q(timestamp, av_get_time_base_q(), stream_time_base);
}

void StreamProcessor::set_decoder(
    const AVCodecParameters* codecpar,
    const c10::optional<std::string>& decoder_name,
    const c10::optional<OptionDict>& decoder_option,
    const torch::Device& device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!codec_ctx, "Decoder has already been set.");
  codec_ctx = get_codec_ctx(codecpar, decoder_name, decoder_option, device);
}

////////////////////////////////////////////////////////////////////////////////
// Query methods
////////////////////////////////////////////////////////////////////////////////
std::string StreamProcessor::get_filter_description(KeyType key) const {
  return post_processes.at(key)->get_filter_desc();
}

FilterGraphOutputInfo StreamProcessor::get_filter_output_info(
    KeyType key) const {
  return post_processes.at(key)->get_filter_output_info();
}

bool StreamProcessor::is_buffer_ready() const {
  for (const auto& it : post_processes) {
    if (!it.second->is_buffer_ready()) {
      return false;
    }
  }
  return true;
}

bool StreamProcessor::is_decoder_set() const {
  return codec_ctx;
}

////////////////////////////////////////////////////////////////////////////////
// The streaming process
////////////////////////////////////////////////////////////////////////////////
// 0: some kind of success
// <0: Some error happened
int StreamProcessor::process_packet(AVPacket* packet) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      is_decoder_set(),
      "Decoder must have been set prior to calling this function.");
  int ret = avcodec_send_packet(codec_ctx, packet);
  while (ret >= 0) {
    ret = avcodec_receive_frame(codec_ctx, frame);
    //  AVERROR(EAGAIN) means that new input data is required to return new
    //  output.
    if (ret == AVERROR(EAGAIN))
      return 0;
    if (ret == AVERROR_EOF)
      return send_frame(nullptr);
    if (ret < 0)
      return ret;

    // If pts is undefined then overwrite with best effort estimate.
    // In this case, best_effort_timestamp is basically the number of frames
    // emit from decoder.
    //
    // We need valid pts because filter_graph does not fall back to
    // best_effort_timestamp.
    if (frame->pts == AV_NOPTS_VALUE) {
      if (frame->best_effort_timestamp == AV_NOPTS_VALUE) {
        // This happens in drain mode.
        // When the decoder enters drain mode, it starts flushing the internally
        // buffered frames, of which PTS cannot be estimated.
        //
        // This is because they might be intra-frames not in chronological
        // order. In this case, we use received frames as-is in the order they
        // are received.
        frame->pts = codec_ctx->frame_number + 1;
      } else {
        frame->pts = frame->best_effort_timestamp;
      }
    }

    // When the value of discard_before_pts is 0, we consider that the seek is
    // not performed and all the frames are passed to downstream
    // unconditionally.
    //
    // Two reasons for this behavior;
    // 1. When seek mode is not precise, we do not discard any frame.
    //    In this case discard_before_pts is set to zero.
    // 2. When users seek to zero, what they expect is to get to the beginning
    //    of the data.
    //
    // Note: discard_before_pts < 0 is UB.
    if (discard_before_pts <= 0 || frame->pts >= discard_before_pts) {
      send_frame(frame);
    }

    // else we can just unref the frame and continue
    av_frame_unref(frame);
  }
  return ret;
}

void StreamProcessor::flush() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      is_decoder_set(),
      "Decoder must have been set prior to calling this function.");
  avcodec_flush_buffers(codec_ctx);
  for (auto& ite : post_processes) {
    ite.second->flush();
  }
}

// 0: some kind of success
// <0: Some error happened
int StreamProcessor::send_frame(AVFrame* frame_) {
  int ret = 0;
  for (auto& ite : post_processes) {
    int ret2 = ite.second->process_frame(frame_);
    if (ret2 < 0)
      ret = ret2;
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Retrieval
////////////////////////////////////////////////////////////////////////////////
c10::optional<Chunk> StreamProcessor::pop_chunk(KeyType key) {
  return post_processes.at(key)->pop_chunk();
}

} // namespace torchaudio::io
