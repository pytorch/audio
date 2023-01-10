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
    TORCH_CHECK(pCodec, ss.str());
  }

  AVCodecContext* pCodecContext = avcodec_alloc_context3(pCodec);
  TORCH_CHECK(pCodecContext, "Failed to allocate CodecContext.");
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
  TORCH_CHECK(
      false,
      "CUDA device was requested, but the codec \"",
      pCodec->name,
      "\" is not supported.");
}
#endif

void init_codec_context(
    AVCodecContext* pCodecContext,
    AVCodecParameters* pParams,
    const c10::optional<OptionDict>& decoder_option,
    const torch::Device& device,
    enum AVPixelFormat* pHwFmt) {
  int ret = avcodec_parameters_to_context(pCodecContext, pParams);
  TORCH_CHECK(
      ret >= 0, "Failed to set CodecContext parameter: " + av_err2string(ret));

#ifdef USE_CUDA
  // Enable HW Acceleration
  if (device.type() == c10::DeviceType::CUDA) {
    *pHwFmt = get_cuda_config(pCodecContext->codec)->pix_fmt;
    // https://www.ffmpeg.org/doxygen/trunk/hw__decode_8c_source.html#l00221
    // 1. Set HW pixel format (config->pix_fmt) to opaue pointer.
    pCodecContext->opaque = static_cast<void*>(pHwFmt);
    // 2. Set pCodecContext->get_format call back function which
    // will retrieve the HW pixel format from opaque pointer.
    pCodecContext->get_format = get_hw_format;
  }
#endif

  AVDictionary* opts = get_option_dict(decoder_option);

  // Default to single thread execution.
  if (!av_dict_get(opts, "threads", nullptr, 0)) {
    av_dict_set(&opts, "threads", "1", 0);
  }

  ret = avcodec_open2(pCodecContext, pCodecContext->codec, &opts);
  clean_up_dict(opts);

  TORCH_CHECK(
      ret >= 0, "Failed to initialize CodecContext: " + av_err2string(ret));

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
  init_codec_context(pCodecContext, pParam, decoder_option, device, &pHwFmt);
}

int Decoder::process_packet(AVPacket* pPacket) {
  return avcodec_send_packet(pCodecContext, pPacket);
}

int Decoder::get_frame(AVFrame* pFrame) {
  return avcodec_receive_frame(pCodecContext, pFrame);
}

int Decoder::get_frame_number() const {
  return pCodecContext->frame_number;
}

void Decoder::flush_buffer() {
  avcodec_flush_buffers(pCodecContext);
}

} // namespace ffmpeg
} // namespace torchaudio
