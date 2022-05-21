#include <c10/util/Exception.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace torchaudio {
namespace ffmpeg {

////////////////////////////////////////////////////////////////////////////////
// AVFormatContext
////////////////////////////////////////////////////////////////////////////////
void AVFormatContextDeleter::operator()(AVFormatContext* p) {
  avformat_close_input(&p);
};

namespace {

AVDictionary* get_option_dict(const OptionDict& option) {
  AVDictionary* opt = nullptr;
  for (const auto& it : option) {
    av_dict_set(&opt, it.first.c_str(), it.second.c_str(), 0);
  }
  return opt;
}

std::vector<std::string> clean_up_dict(AVDictionary* p) {
  std::vector<std::string> ret;

  // Check and copy unused keys, clean up the original dictionary
  AVDictionaryEntry* t = nullptr;
  do {
    t = av_dict_get(p, "", t, AV_DICT_IGNORE_SUFFIX);
    if (t) {
      ret.emplace_back(t->key);
    }
  } while (t);
  av_dict_free(&p);
  return ret;
}

std::string join(std::vector<std::string> vars) {
  std::stringstream ks;
  for (size_t i = 0; i < vars.size(); ++i) {
    if (i == 0) {
      ks << "\"" << vars[i] << "\"";
    } else {
      ks << ", \"" << vars[i] << "\"";
    }
  }
  return ks.str();
}

// https://github.com/FFmpeg/FFmpeg/blob/4e6debe1df7d53f3f59b37449b82265d5c08a172/doc/APIchanges#L252-L260
// Starting from libavformat 59 (ffmpeg 5),
// AVInputFormat is const and related functions expect constant.
#if LIBAVFORMAT_VERSION_MAJOR >= 59
#define AVINPUT_FORMAT_CONST const
#else
#define AVINPUT_FORMAT_CONST
#endif

} // namespace

AVFormatContextPtr get_input_format_context(
    const std::string& src,
    const c10::optional<std::string>& device,
    const OptionDict& option,
    AVIOContext* io_ctx) {
  AVFormatContext* pFormat = avformat_alloc_context();
  if (!pFormat) {
    throw std::runtime_error("Failed to allocate AVFormatContext.");
  }
  if (io_ctx) {
    pFormat->pb = io_ctx;
  }

  auto* pInput = [&]() -> AVINPUT_FORMAT_CONST AVInputFormat* {
    if (device.has_value()) {
      std::string device_str = device.value();
      AVINPUT_FORMAT_CONST AVInputFormat* p =
          av_find_input_format(device_str.c_str());
      if (!p) {
        std::ostringstream msg;
        msg << "Unsupported device/format: \"" << device_str << "\"";
        throw std::runtime_error(msg.str());
      }
      return p;
    }
    return nullptr;
  }();

  AVDictionary* opt = get_option_dict(option);
  int ret = avformat_open_input(&pFormat, src.c_str(), pInput, &opt);

  auto unused_keys = clean_up_dict(opt);

  if (unused_keys.size()) {
    throw std::runtime_error("Unexpected options: " + join(unused_keys));
  }

  if (ret < 0)
    throw std::runtime_error(
        "Failed to open the input \"" + src + "\" (" + av_err2string(ret) +
        ").");
  return AVFormatContextPtr(pFormat);
}

AVFormatContextPtr::AVFormatContextPtr(AVFormatContext* p)
    : Wrapper<AVFormatContext, AVFormatContextDeleter>(p) {}

////////////////////////////////////////////////////////////////////////////////
// AVIO
////////////////////////////////////////////////////////////////////////////////
void AVIOContextDeleter::operator()(AVIOContext* p) {
  av_freep(&p->buffer);
  av_freep(&p);
};

AVIOContextPtr::AVIOContextPtr(AVIOContext* p)
    : Wrapper<AVIOContext, AVIOContextDeleter>(p) {}

////////////////////////////////////////////////////////////////////////////////
// AVPacket
////////////////////////////////////////////////////////////////////////////////
void AVPacketDeleter::operator()(AVPacket* p) {
  av_packet_free(&p);
};

namespace {
AVPacket* get_av_packet() {
  AVPacket* pPacket = av_packet_alloc();
  if (!pPacket)
    throw std::runtime_error("Failed to allocate AVPacket object.");
  return pPacket;
}
} // namespace

AVPacketPtr::AVPacketPtr()
    : Wrapper<AVPacket, AVPacketDeleter>(get_av_packet()) {}

////////////////////////////////////////////////////////////////////////////////
// AVPacket - buffer unref
////////////////////////////////////////////////////////////////////////////////
AutoPacketUnref::AutoPacketUnref(AVPacketPtr& p) : p_(p){};
AutoPacketUnref::~AutoPacketUnref() {
  av_packet_unref(p_);
}
AutoPacketUnref::operator AVPacket*() const {
  return p_;
}

////////////////////////////////////////////////////////////////////////////////
// AVFrame
////////////////////////////////////////////////////////////////////////////////
void AVFrameDeleter::operator()(AVFrame* p) {
  av_frame_free(&p);
};
namespace {
AVFrame* get_av_frame() {
  AVFrame* pFrame = av_frame_alloc();
  if (!pFrame)
    throw std::runtime_error("Failed to allocate AVFrame object.");
  return pFrame;
}
} // namespace

AVFramePtr::AVFramePtr() : Wrapper<AVFrame, AVFrameDeleter>(get_av_frame()) {}

////////////////////////////////////////////////////////////////////////////////
// AVCodecContext
////////////////////////////////////////////////////////////////////////////////
void AVCodecContextDeleter::operator()(AVCodecContext* p) {
  avcodec_free_context(&p);
};

namespace {
const AVCodec* get_decode_codec(
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
  return pCodec;
}

} // namespace

AVCodecContextPtr get_decode_context(
    enum AVCodecID codec_id,
    const c10::optional<std::string>& decoder_name) {
  const AVCodec* pCodec = get_decode_codec(codec_id, decoder_name);

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
    const c10::optional<std::string>& decoder_name,
    const OptionDict& decoder_option,
    const torch::Device& device,
    AVBufferRefPtr& pHWBufferRef) {
  const AVCodec* pCodec = get_decode_codec(pParams->codec_id, decoder_name);

  if (avcodec_parameters_to_context(pCodecContext, pParams) < 0) {
    throw std::runtime_error("Failed to set CodecContext parameter.");
  }

#ifdef USE_CUDA
  // Enable HW Acceleration
  if (device.type() == c10::DeviceType::CUDA) {
    const AVCodecHWConfig* config = get_cuda_config(pCodec);
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
    // TODO: check how to deallocate the context
    int err = av_hwdevice_ctx_create(
        &hw_device_ctx,
        AV_HWDEVICE_TYPE_CUDA,
        std::to_string(device.index()).c_str(),
        nullptr,
        0);
    if (err < 0) {
      throw std::runtime_error(
          "Failed to create CUDA device context: " + av_err2string(err));
    }
    assert(hw_device_ctx);
    pCodecContext->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    pHWBufferRef.reset(hw_device_ctx);
  }
#endif

  AVDictionary* opts = get_option_dict(decoder_option);
  if (avcodec_open2(pCodecContext, pCodec, &opts) < 0) {
    throw std::runtime_error("Failed to initialize CodecContext.");
  }
  auto unused_keys = clean_up_dict(opts);
  if (unused_keys.size()) {
    throw std::runtime_error(
        "Unexpected decoder options: " + join(unused_keys));
  }

  if (pParams->codec_type == AVMEDIA_TYPE_AUDIO && !pParams->channel_layout)
    pParams->channel_layout =
        av_get_default_channel_layout(pCodecContext->channels);
}

AVCodecContextPtr::AVCodecContextPtr(AVCodecContext* p)
    : Wrapper<AVCodecContext, AVCodecContextDeleter>(p) {}

////////////////////////////////////////////////////////////////////////////////
// AVBufferRefPtr
////////////////////////////////////////////////////////////////////////////////
void AutoBufferUnref::operator()(AVBufferRef* p) {
  av_buffer_unref(&p);
}

AVBufferRefPtr::AVBufferRefPtr()
    : Wrapper<AVBufferRef, AutoBufferUnref>(nullptr) {}

void AVBufferRefPtr::reset(AVBufferRef* p) {
  TORCH_CHECK(
      !ptr,
      "InternalError: A valid AVBufferRefPtr is being reset. Please file an issue.");
  ptr.reset(p);
}

////////////////////////////////////////////////////////////////////////////////
// AVFilterGraph
////////////////////////////////////////////////////////////////////////////////
void AVFilterGraphDeleter::operator()(AVFilterGraph* p) {
  avfilter_graph_free(&p);
};

namespace {
AVFilterGraph* get_filter_graph() {
  AVFilterGraph* ptr = avfilter_graph_alloc();
  if (!ptr)
    throw std::runtime_error("Failed to allocate resouce.");
  return ptr;
}
} // namespace
AVFilterGraphPtr::AVFilterGraphPtr()
    : Wrapper<AVFilterGraph, AVFilterGraphDeleter>(get_filter_graph()) {}

void AVFilterGraphPtr::reset() {
  ptr.reset(get_filter_graph());
}
} // namespace ffmpeg
} // namespace torchaudio
