#include <c10/util/Exception.h>
#include <libtorio/ffmpeg/ffmpeg.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace torio::io {

////////////////////////////////////////////////////////////////////////////////
// AVDictionary
////////////////////////////////////////////////////////////////////////////////
AVDictionary* get_option_dict(const c10::optional<OptionDict>& option) {
  AVDictionary* opt = nullptr;
  if (option) {
    for (auto const& [key, value] : option.value()) {
      av_dict_set(&opt, key.c_str(), value.c_str(), 0);
    }
  }
  return opt;
}

void clean_up_dict(AVDictionary* p) {
  if (p) {
    std::vector<std::string> unused_keys;
    // Check and copy unused keys, clean up the original dictionary
    AVDictionaryEntry* t = nullptr;
    while ((t = av_dict_get(p, "", t, AV_DICT_IGNORE_SUFFIX))) {
      unused_keys.emplace_back(t->key);
    }
    av_dict_free(&p);
    TORCH_CHECK(
        unused_keys.empty(),
        "Unexpected options: ",
        c10::Join(", ", unused_keys));
  }
}

////////////////////////////////////////////////////////////////////////////////
// AVFormatContext
////////////////////////////////////////////////////////////////////////////////
void AVFormatInputContextDeleter::operator()(AVFormatContext* p) {
  avformat_close_input(&p);
};

AVFormatInputContextPtr::AVFormatInputContextPtr(AVFormatContext* p)
    : Wrapper<AVFormatContext, AVFormatInputContextDeleter>(p) {}

void AVFormatOutputContextDeleter::operator()(AVFormatContext* p) {
  avformat_free_context(p);
};

AVFormatOutputContextPtr::AVFormatOutputContextPtr(AVFormatContext* p)
    : Wrapper<AVFormatContext, AVFormatOutputContextDeleter>(p) {}

////////////////////////////////////////////////////////////////////////////////
// AVIO
////////////////////////////////////////////////////////////////////////////////
void AVIOContextDeleter::operator()(AVIOContext* p) {
  avio_flush(p);
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

AVPacketPtr::AVPacketPtr(AVPacket* p) : Wrapper<AVPacket, AVPacketDeleter>(p) {}

AVPacketPtr alloc_avpacket() {
  AVPacket* p = av_packet_alloc();
  TORCH_CHECK(p, "Failed to allocate AVPacket object.");
  return AVPacketPtr{p};
}

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

AVFramePtr::AVFramePtr(AVFrame* p) : Wrapper<AVFrame, AVFrameDeleter>(p) {}

AVFramePtr alloc_avframe() {
  AVFrame* p = av_frame_alloc();
  TORCH_CHECK(p, "Failed to allocate AVFrame object.");
  return AVFramePtr{p};
};

////////////////////////////////////////////////////////////////////////////////
// AVCodecContext
////////////////////////////////////////////////////////////////////////////////
void AVCodecContextDeleter::operator()(AVCodecContext* p) {
  avcodec_free_context(&p);
};

AVCodecContextPtr::AVCodecContextPtr(AVCodecContext* p)
    : Wrapper<AVCodecContext, AVCodecContextDeleter>(p) {}

////////////////////////////////////////////////////////////////////////////////
// AVBufferRefPtr
////////////////////////////////////////////////////////////////////////////////
void AutoBufferUnref::operator()(AVBufferRef* p) {
  av_buffer_unref(&p);
}

AVBufferRefPtr::AVBufferRefPtr(AVBufferRef* p)
    : Wrapper<AVBufferRef, AutoBufferUnref>(p) {}

////////////////////////////////////////////////////////////////////////////////
// AVFilterGraph
////////////////////////////////////////////////////////////////////////////////
void AVFilterGraphDeleter::operator()(AVFilterGraph* p) {
  avfilter_graph_free(&p);
};

AVFilterGraphPtr::AVFilterGraphPtr(AVFilterGraph* p)
    : Wrapper<AVFilterGraph, AVFilterGraphDeleter>(p) {}

////////////////////////////////////////////////////////////////////////////////
// AVCodecParameters
////////////////////////////////////////////////////////////////////////////////
void AVCodecParametersDeleter::operator()(AVCodecParameters* codecpar) {
  avcodec_parameters_free(&codecpar);
}

AVCodecParametersPtr::AVCodecParametersPtr(AVCodecParameters* p)
    : Wrapper<AVCodecParameters, AVCodecParametersDeleter>(p) {}

} // namespace torio::io
