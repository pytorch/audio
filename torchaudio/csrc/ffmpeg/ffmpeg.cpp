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

AVDictionary* get_option_dict(
    const std::map<std::string, std::string>& option) {
  AVDictionary* opt = nullptr;
  for (auto& it : option) {
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

AVFormatContext* get_format_context(
    const std::string& src,
    const std::string& device,
    const std::map<std::string, std::string>& option) {
  AVFormatContext* pFormat = NULL;
  AVInputFormat* pInput =
      device.empty() ? NULL : av_find_input_format(device.c_str());
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
  return pFormat;
}
} // namespace

AVFormatContextPtr::AVFormatContextPtr(
    const std::string& src,
    const std::string& device,
    const std::map<std::string, std::string>& option)
    : Wrapper<AVFormatContext, AVFormatContextDeleter>(
          get_format_context(src, device, option)) {
  if (avformat_find_stream_info(ptr.get(), NULL) < 0)
    throw std::runtime_error("Failed to find stream information.");
}

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
AVCodecContext* get_codec_context(AVCodecParameters* pParams) {
  const AVCodec* pCodec = avcodec_find_decoder(pParams->codec_id);

  if (!pCodec) {
    throw std::runtime_error("Unknown codec.");
  }

  AVCodecContext* pCodecContext = avcodec_alloc_context3(pCodec);
  if (!pCodecContext) {
    throw std::runtime_error("Failed to allocate CodecContext.");
  }
  return pCodecContext;
}

void init_codec_context(
    AVCodecContext* pCodecContext,
    AVCodecParameters* pParams) {
  const AVCodec* pCodec = avcodec_find_decoder(pParams->codec_id);

  if (avcodec_parameters_to_context(pCodecContext, pParams) < 0) {
    throw std::runtime_error("Failed to set CodecContext parameter.");
  }

  if (avcodec_open2(pCodecContext, pCodec, NULL) < 0) {
    throw std::runtime_error("Failed to initialize CodecContext.");
  }

  if (pParams->codec_type == AVMEDIA_TYPE_AUDIO && !pParams->channel_layout)
    pParams->channel_layout =
        av_get_default_channel_layout(pCodecContext->channels);
}
} // namespace

AVCodecContextPtr::AVCodecContextPtr(AVCodecParameters* pParam)
    : Wrapper<AVCodecContext, AVCodecContextDeleter>(
          get_codec_context(pParam)) {
  init_codec_context(ptr.get(), pParam);
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
} // namespace ffmpeg
} // namespace torchaudio
