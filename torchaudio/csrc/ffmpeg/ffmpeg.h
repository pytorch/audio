// One stop header for all ffmepg needs
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/pixdesc.h>
}

namespace torchaudio {
namespace ffmpeg {

// Base structure that handles memory management.
// Resource is freed by the destructor of unique_ptr,
// which will call custom delete mechanism provided via Deleter
// https://stackoverflow.com/a/19054280
//
// The resource allocation will be provided by custom constructors.
template <typename T, typename Deleter>
class Wrapper {
 protected:
  std::unique_ptr<T, Deleter> ptr;

 public:
  Wrapper() = delete;
  Wrapper<T, Deleter>(T* t) : ptr(t){};
  T* operator->() const {
    return ptr.get();
  };
  explicit operator bool() const {
    return (bool)ptr;
  };
  operator T*() const {
    return ptr.get();
  }
};

////////////////////////////////////////////////////////////////////////////////
// AVFormatContext
////////////////////////////////////////////////////////////////////////////////
struct AVFormatContextDeleter {
  void operator()(AVFormatContext* p);
};

struct AVFormatContextPtr
    : public Wrapper<AVFormatContext, AVFormatContextDeleter> {
  AVFormatContextPtr(
      const std::string& src,
      const std::string& device,
      const std::map<std::string, std::string>& option);
};

////////////////////////////////////////////////////////////////////////////////
// AVPacket
////////////////////////////////////////////////////////////////////////////////
struct AVPacketDeleter {
  void operator()(AVPacket* p);
};

struct AVPacketPtr : public Wrapper<AVPacket, AVPacketDeleter> {
  AVPacketPtr();
};

////////////////////////////////////////////////////////////////////////////////
// AVPacket - buffer unref
////////////////////////////////////////////////////////////////////////////////
// AVPacket structure employs two-staged memory allocation.
// The first-stage is for allocating AVPacket object itself, and it typically
// happens only once throughout the lifetime of application.
// The second-stage is for allocating the content (media data) each time the
// input file is processed and a chunk of data is read. The memory allocated
// during this time has to be released before the next iteration.
// The first-stage memory management is handled by `AVPacketPtr`.
// `AutoPacketUnref` handles the second-stage memory management.
struct AutoPacketUnref {
  AVPacketPtr& p_;
  AutoPacketUnref(AVPacketPtr& p);
  ~AutoPacketUnref();
  operator AVPacket*() const;
};

////////////////////////////////////////////////////////////////////////////////
// AVFrame
////////////////////////////////////////////////////////////////////////////////
struct AVFrameDeleter {
  void operator()(AVFrame* p);
};

struct AVFramePtr : public Wrapper<AVFrame, AVFrameDeleter> {
  AVFramePtr();
};

////////////////////////////////////////////////////////////////////////////////
// AVCodecContext
////////////////////////////////////////////////////////////////////////////////
struct AVCodecContextDeleter {
  void operator()(AVCodecContext* p);
};
struct AVCodecContextPtr
    : public Wrapper<AVCodecContext, AVCodecContextDeleter> {
  AVCodecContextPtr(AVCodecParameters* pParam);
};

////////////////////////////////////////////////////////////////////////////////
// AVFilterGraph
////////////////////////////////////////////////////////////////////////////////
struct AVFilterGraphDeleter {
  void operator()(AVFilterGraph* p);
};
struct AVFilterGraphPtr : public Wrapper<AVFilterGraph, AVFilterGraphDeleter> {
  AVFilterGraphPtr();
};
} // namespace ffmpeg
} // namespace torchaudio
