// One stop header for all ffmepg needs
#pragma once
#include <torch/types.h>
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
#include <libavformat/avio.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/pixdesc.h>
}

/// @cond

namespace torio {
namespace io {

using OptionDict = std::map<std::string, std::string>;

// https://github.com/FFmpeg/FFmpeg/blob/4e6debe1df7d53f3f59b37449b82265d5c08a172/doc/APIchanges#L252-L260
// Starting from libavformat 59 (ffmpeg 5),
// AVInputFormat is const and related functions expect constant.
#if LIBAVFORMAT_VERSION_MAJOR >= 59
#define AVFORMAT_CONST const
#else
#define AVFORMAT_CONST
#endif

// Replacement of av_err2str, which causes
// `error: taking address of temporary array`
// https://github.com/joncampbell123/composite-video-simulator/issues/5
av_always_inline std::string av_err2string(int errnum) {
  char str[AV_ERROR_MAX_STRING_SIZE];
  return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}

// Base structure that handles memory management.
// Resource is freed by the destructor of unique_ptr,
// which will call custom delete mechanism provided via Deleter
// https://stackoverflow.com/a/19054280
//
// The resource allocation will be provided by custom constructors.
template <typename T, typename Deleter>
class Wrapper {
  std::unique_ptr<T, Deleter> ptr;

 public:
  Wrapper() = delete;
  explicit Wrapper<T, Deleter>(T* t) : ptr(t) {}
  T* operator->() const {
    return ptr.get();
  }
  explicit operator bool() const {
    return (bool)ptr;
  }
  operator T*() const {
    return ptr.get();
  }
};

////////////////////////////////////////////////////////////////////////////////
// AVDictionary
////////////////////////////////////////////////////////////////////////////////
// Since AVDictionaries are relocated by FFmpeg APIs it does not suit to
// IIRC-semantic. Instead we provide helper functions.

// Convert standard dict to FFmpeg native type
AVDictionary* get_option_dict(const std::optional<OptionDict>& option);

// Clean up the dict after use. If there is an unsed key, throw runtime error
void clean_up_dict(AVDictionary* p);

////////////////////////////////////////////////////////////////////////////////
// AVFormatContext
////////////////////////////////////////////////////////////////////////////////
struct AVFormatInputContextDeleter {
  void operator()(AVFormatContext* p);
};

struct AVFormatInputContextPtr
    : public Wrapper<AVFormatContext, AVFormatInputContextDeleter> {
  explicit AVFormatInputContextPtr(AVFormatContext* p);
};

struct AVFormatOutputContextDeleter {
  void operator()(AVFormatContext* p);
};

struct AVFormatOutputContextPtr
    : public Wrapper<AVFormatContext, AVFormatOutputContextDeleter> {
  explicit AVFormatOutputContextPtr(AVFormatContext* p);
};

////////////////////////////////////////////////////////////////////////////////
// AVIO
////////////////////////////////////////////////////////////////////////////////
struct AVIOContextDeleter {
  void operator()(AVIOContext* p);
};

struct AVIOContextPtr : public Wrapper<AVIOContext, AVIOContextDeleter> {
  explicit AVIOContextPtr(AVIOContext* p);
};

////////////////////////////////////////////////////////////////////////////////
// AVPacket
////////////////////////////////////////////////////////////////////////////////
struct AVPacketDeleter {
  void operator()(AVPacket* p);
};

struct AVPacketPtr : public Wrapper<AVPacket, AVPacketDeleter> {
  explicit AVPacketPtr(AVPacket* p);
};

AVPacketPtr alloc_avpacket();

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
  explicit AutoPacketUnref(AVPacketPtr& p);
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
  explicit AVFramePtr(AVFrame* p);
};

AVFramePtr alloc_avframe();

////////////////////////////////////////////////////////////////////////////////
// AutoBufferUnrer is responsible for performing unref at the end of lifetime
// of AVBufferRefPtr.
////////////////////////////////////////////////////////////////////////////////
struct AutoBufferUnref {
  void operator()(AVBufferRef* p);
};

struct AVBufferRefPtr : public Wrapper<AVBufferRef, AutoBufferUnref> {
  explicit AVBufferRefPtr(AVBufferRef* p);
};

////////////////////////////////////////////////////////////////////////////////
// AVCodecContext
////////////////////////////////////////////////////////////////////////////////
struct AVCodecContextDeleter {
  void operator()(AVCodecContext* p);
};
struct AVCodecContextPtr
    : public Wrapper<AVCodecContext, AVCodecContextDeleter> {
  explicit AVCodecContextPtr(AVCodecContext* p);
};

////////////////////////////////////////////////////////////////////////////////
// AVFilterGraph
////////////////////////////////////////////////////////////////////////////////
struct AVFilterGraphDeleter {
  void operator()(AVFilterGraph* p);
};
struct AVFilterGraphPtr : public Wrapper<AVFilterGraph, AVFilterGraphDeleter> {
  explicit AVFilterGraphPtr(AVFilterGraph* p);
};

////////////////////////////////////////////////////////////////////////////////
// AVCodecParameters
////////////////////////////////////////////////////////////////////////////////
struct AVCodecParametersDeleter {
  void operator()(AVCodecParameters* p);
};

struct AVCodecParametersPtr
    : public Wrapper<AVCodecParameters, AVCodecParametersDeleter> {
  explicit AVCodecParametersPtr(AVCodecParameters* p);
};

struct StreamParams {
  AVCodecParametersPtr codec_params{nullptr};
  AVRational time_base{};
  int stream_index{};
};
} // namespace io
} // namespace torio

/// @endcond
