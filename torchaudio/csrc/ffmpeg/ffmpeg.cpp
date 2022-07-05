#include <c10/util/Exception.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace torchaudio {
namespace ffmpeg {

////////////////////////////////////////////////////////////////////////////////
// AVDictionary
////////////////////////////////////////////////////////////////////////////////

AVDictionary* get_option_dict(const OptionDict& option) {
  AVDictionary* opt = nullptr;
  for (const auto& it : option) {
    av_dict_set(&opt, it.first.c_str(), it.second.c_str(), 0);
  }
  return opt;
}

void clean_up_dict(AVDictionary* p) {
  std::vector<std::string> unused_keys;
  // Check and copy unused keys, clean up the original dictionary
  AVDictionaryEntry* t = nullptr;
  while ((t = av_dict_get(p, "", t, AV_DICT_IGNORE_SUFFIX))) {
    unused_keys.emplace_back(t->key);
  }
  av_dict_free(&p);

  if (!unused_keys.empty()) {
    throw std::runtime_error(
        "Unexpected options: " + c10::Join(", ", unused_keys));
  }
}

////////////////////////////////////////////////////////////////////////////////
// AVFormatContext
////////////////////////////////////////////////////////////////////////////////
void AVFormatContextDeleter::operator()(AVFormatContext* p) {
  avformat_close_input(&p);
};

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
