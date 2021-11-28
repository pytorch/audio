#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace ffmpeg {

////////////////////////////////////////////////////////////////////////////////
// AVFormatContext
////////////////////////////////////////////////////////////////////////////////
void AVFormatContextDeleter::operator()(AVFormatContext* p) {
  avformat_close_input(&p);
};

namespace {
AVFormatContext* get_format_context(const std::string& src) {
  AVFormatContext* pFormat = NULL;
  if (avformat_open_input(&pFormat, src.c_str(), NULL, NULL) < 0)
    throw std::runtime_error("Failed to open the input: " + src);
  return pFormat;
}
} // namespace

AVFormatContextPtr::AVFormatContextPtr(const std::string& src)
    : Wrapper<AVFormatContext, AVFormatContextDeleter>(
          get_format_context(src)) {
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

///////////////////////////////////////////////////////////////////////////////
// AVFrame - buffer unref
////////////////////////////////////////////////////////////////////////////////
AutoFrameUnref::AutoFrameUnref(AVFramePtr& p) : p_(p){};
AutoFrameUnref::~AutoFrameUnref() {
  av_frame_unref(p_);
}
AutoFrameUnref::operator AVFrame*() const {
  return p_;
}

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
