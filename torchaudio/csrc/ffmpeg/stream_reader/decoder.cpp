#include <torchaudio/csrc/ffmpeg/stream_reader/decoder.h>

namespace torchaudio {
namespace ffmpeg {

////////////////////////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////////////////////////
namespace {
AVCodecContextPtr get_decode_context(
    enum AVCodecID codec_id,
    const c10::optional<std::string>& decoder_name) {
  const AVCodec* pCodec = !decoder_name.has_value()
      ? avcodec_find_decoder(codec_id)
      : avcodec_find_decoder_by_name(decoder_name.value().c_str());

  if (!pCodec) {
    std::stringstream ss;
    if (!decoder_name.has_value()) {
      ss << "Unsupported codec: \"" << avcodec_get_name(codec_id) << "\", ("
         << codec_id << ").";
    } else {
      ss << "Unsupported codec: \"" << decoder_name.value() << "\".";
    }
    throw std::runtime_error(ss.str());
  }

  AVCodecContext* pCodecContext = avcodec_alloc_context3(pCodec);
  if (!pCodecContext) {
    throw std::runtime_error("Failed to allocate CodecContext.");
  }
  return AVCodecContextPtr(pCodecContext);
}

#ifdef USE_CUDA
enum AVPixelFormat get_hw_format(
    AVCodecContext* ctx,
    const enum AVPixelFormat* pix_fmts) {
  const enum AVPixelFormat* p = nullptr;
  AVPixelFormat pix_fmt = *static_cast<AVPixelFormat*>(ctx->opaque);
  for (p = pix_fmts; *p != -1; p++) {
    if (*p == pix_fmt) {
      return *p;
    }
  }
  TORCH_WARN("Failed to get HW surface format.");
  return AV_PIX_FMT_NONE;
}

const AVCodecHWConfig* get_cuda_config(const AVCodec* pCodec) {
  for (int i = 0;; ++i) {
    const AVCodecHWConfig* config = avcodec_get_hw_config(pCodec, i);
    if (!config) {
      break;
    }
    if (config->device_type == AV_HWDEVICE_TYPE_CUDA &&
        config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
      return config;
    }
  }
  std::stringstream ss;
  ss << "CUDA device was requested, but the codec \"" << pCodec->name
     << "\" is not supported.";
  throw std::runtime_error(ss.str());
}
#endif

void init_codec_context(
    AVCodecContext* pCodecContext,
    AVCodecParameters* pParams,
    const c10::optional<OptionDict>& decoder_option,
    const torch::Device& device,
    AVBufferRefPtr& pHWBufferRef) {
  int ret = avcodec_parameters_to_context(pCodecContext, pParams);
  if (ret < 0) {
    throw std::runtime_error(
        "Failed to set CodecContext parameter: " + av_err2string(ret));
  }

#ifdef USE_CUDA
  // Enable HW Acceleration
  if (device.type() == c10::DeviceType::CUDA) {
    const AVCodecHWConfig* config = get_cuda_config(pCodecContext->codec);
    // TODO: check how to log
    // C10_LOG << "Decoder " << pCodec->name << " supports device " <<
    // av_hwdevice_get_type_name(config->device_type);

    // https://www.ffmpeg.org/doxygen/trunk/hw__decode_8c_source.html#l00221
    // 1. Set HW pixel format (config->pix_fmt) to opaue pointer.
    static thread_local AVPixelFormat pix_fmt = config->pix_fmt;
    pCodecContext->opaque = static_cast<void*>(&pix_fmt);
    // 2. Set pCodecContext->get_format call back function which
    // will retrieve the HW pixel format from opaque pointer.
    pCodecContext->get_format = get_hw_format;
    // 3. Create HW device context and set to pCodecContext.
    AVBufferRef* hw_device_ctx = nullptr;
    ret = av_hwdevice_ctx_create(
        &hw_device_ctx,
        AV_HWDEVICE_TYPE_CUDA,
        std::to_string(device.index()).c_str(),
        nullptr,
        0);
    if (ret < 0) {
      throw std::runtime_error(
          "Failed to create CUDA device context: " + av_err2string(ret));
    }
    assert(hw_device_ctx);
    pCodecContext->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    pHWBufferRef.reset(hw_device_ctx);
  }
#endif

  AVDictionary* opts = get_option_dict(decoder_option);
  ret = avcodec_open2(pCodecContext, pCodecContext->codec, &opts);
  clean_up_dict(opts);

  if (ret < 0) {
    throw std::runtime_error(
        "Failed to initialize CodecContext: " + av_err2string(ret));
  }

  if (pParams->codec_type == AVMEDIA_TYPE_AUDIO && !pParams->channel_layout)
    pParams->channel_layout =
        av_get_default_channel_layout(pCodecContext->channels);
}
} // namespace

Decoder::Decoder(
    AVCodecParameters* pParam,
    const c10::optional<std::string>& decoder_name,
    const c10::optional<OptionDict>& decoder_option,
    const torch::Device& device)
    : pCodecContext(get_decode_context(pParam->codec_id, decoder_name)) {
  init_codec_context(
      pCodecContext, pParam, decoder_option, device, pHWBufferRef);
}

int Decoder::process_packet(AVPacket* pPacket) {
  return avcodec_send_packet(pCodecContext, pPacket);
}

int Decoder::get_frame(AVFrame* pFrame) {
  return avcodec_receive_frame(pCodecContext, pFrame);
}

void Decoder::flush_buffer() {
  avcodec_flush_buffers(pCodecContext);
}

} // namespace ffmpeg
} // namespace torchaudio
