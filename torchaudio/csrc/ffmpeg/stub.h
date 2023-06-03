#pragma once

// Abstraction of the access to FFmpeg libraries.
//
// Do not include this in header files.
// Include this header in implementation files and prepend
// all the calls to libav functions with FFMPEG macro.
//
// If DLOPEN_FFMPEG is not defined, FFMPEG macro is empty.
// In this case, FFmpeg libraries are linked at the time torchaudio is built.
//
// If DLOPEN_FFMPEG is defined, FFMPEG macro becomes a function call to
// fetch a stub instance of FFmpeg libraries.
// This function also initializes the function pointers by automatically
// dlopens all the required libraries.
//

#ifndef DLOPEN_FFMPEG
#define FFMPEG
#else
#define FFMPEG detail::ffmpeg_stub().

#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio::io::detail {

struct FFmpegStub;

// dlopen FFmpeg libraries and populate the methods of stub instance,
// then return the reference to the stub instance
FFmpegStub& ffmpeg_stub();

struct FFmpegStub {
  /////////////////////////////////////////////////////////////////////////////
  // libavutil
  /////////////////////////////////////////////////////////////////////////////

  AVBufferRef* (*av_buffer_ref)(const AVBufferRef*);

  void (*av_buffer_unref)(AVBufferRef**);

  AVRational (*av_d2q)(double, int) av_const;

  void (*av_dict_free)(AVDictionary**);

  AVDictionaryEntry* (*av_dict_get)(
      const AVDictionary*,
      const char*,
      const AVDictionaryEntry*,
      int);

  int (*av_dict_set)(AVDictionary**, const char*, const char*, int);

  AVFrame* (*av_frame_alloc)();

  void (*av_frame_free)(AVFrame**);

  int (*av_frame_get_buffer)(AVFrame*, int);

  int (*av_frame_is_writable)(AVFrame*);

  int (*av_frame_make_writable)(AVFrame*);

  void (*av_frame_unref)(AVFrame*);

  void (*av_freep)(void*);

  int (*av_get_channel_layout_nb_channels)(uint64_t);

  const char* (*av_get_channel_name)(uint64_t);

  int64_t (*av_get_default_channel_layout)(int);

  const char* (*av_get_media_type_string)(enum AVMediaType);

  enum AVPixelFormat (*av_get_pix_fmt)(const char*);

  const char* (*av_get_pix_fmt_name)(enum AVPixelFormat);

  enum AVSampleFormat (*av_get_sample_fmt)(const char*);

  const char* (*av_get_sample_fmt_name)(enum AVSampleFormat);

  AVRational (*av_get_time_base_q)();

  int (*av_hwdevice_ctx_create)(
      AVBufferRef**,
      enum AVHWDeviceType,
      const char*,
      AVDictionary*,
      int);

  AVBufferRef* (*av_hwframe_ctx_alloc)(AVBufferRef*);

  int (*av_hwframe_ctx_init)(AVBufferRef*);

  int (*av_hwframe_get_buffer)(AVBufferRef*, AVFrame*, int);

  int (*av_log_get_level)();

  void (*av_log_set_level)(int);

  void* (*av_malloc)(size_t);

  const AVPixFmtDescriptor* (*av_pix_fmt_desc_get)(enum AVPixelFormat);

  int64_t (*av_rescale_q)(int64_t, AVRational, AVRational) av_const;

  int (*av_sample_fmt_is_planar)(enum AVSampleFormat);

  char* (*av_strdup)(const char*);

  int (*av_strerror)(int, char*, size_t);

  unsigned (*avutil_version)();

  /////////////////////////////////////////////////////////////////////////////
  // libavcodec
  /////////////////////////////////////////////////////////////////////////////

  int (*av_codec_is_decoder)(const AVCodec*);

  int (*av_codec_is_encoder)(const AVCodec*);

  const AVCodec* (*av_codec_iterate)(void**);

  AVPacket* (*av_packet_alloc)();

  AVPacket* (*av_packet_clone)(const AVPacket*);

  void (*av_packet_free)(AVPacket**);

  int (*av_packet_ref)(AVPacket*, const AVPacket*);

  void (*av_packet_rescale_ts)(AVPacket*, AVRational, AVRational);

  void (*av_packet_unref)(AVPacket*);

  AVCodecContext* (*avcodec_alloc_context3)(const AVCodec*);

  const char* (*avcodec_configuration)();

  const AVCodecDescriptor* (*avcodec_descriptor_get)(enum AVCodecID);

  AVCodec* (*avcodec_find_decoder)(enum AVCodecID);

  AVCodec* (*avcodec_find_decoder_by_name)(const char*);

  AVCodec* (*avcodec_find_encoder)(enum AVCodecID);

  AVCodec* (*avcodec_find_encoder_by_name)(const char*);

  void (*avcodec_flush_buffers)(AVCodecContext*);

  void (*avcodec_free_context)(AVCodecContext**);

  const AVCodecHWConfig* (*avcodec_get_hw_config)(const AVCodec*, int);

  const char* (*avcodec_get_name)(enum AVCodecID);

  int (*avcodec_open2)(AVCodecContext*, const AVCodec*, AVDictionary**);

  AVCodecParameters* (*avcodec_parameters_alloc)();

  int (*avcodec_parameters_copy)(AVCodecParameters*, const AVCodecParameters*);

  void (*avcodec_parameters_free)(AVCodecParameters**);

  int (*avcodec_parameters_from_context)(
      AVCodecParameters*,
      const AVCodecContext*);

  int (*avcodec_parameters_to_context)(
      AVCodecContext*,
      const AVCodecParameters*);

  int (*avcodec_receive_frame)(AVCodecContext*, AVFrame*);

  int (*avcodec_receive_packet)(AVCodecContext*, AVPacket*);

  int (*avcodec_send_frame)(AVCodecContext*, const AVFrame*);

  int (*avcodec_send_packet)(AVCodecContext*, const AVPacket*);

  unsigned (*avcodec_version)();

  /////////////////////////////////////////////////////////////////////////////
  // libavformat
  /////////////////////////////////////////////////////////////////////////////

  const AVInputFormat* (*av_demuxer_iterate)(void**);

  void (*av_dump_format)(AVFormatContext*, int, const char*, int);

  int (*av_find_best_stream)(
      AVFormatContext*,
      enum AVMediaType,
      int,
      int,
      AVCodec**,
      int);

  AVInputFormat* (*av_find_input_format)(const char*);

  AVRational (*av_guess_frame_rate)(AVFormatContext*, AVStream*, AVFrame*);

  int (*av_interleaved_write_frame)(AVFormatContext*, AVPacket*);

  const AVOutputFormat* (*av_muxer_iterate)(void**);

  int (*av_read_frame)(AVFormatContext*, AVPacket*);

  int (*av_seek_frame)(AVFormatContext*, int, int64_t, int);

  int (*av_write_trailer)(AVFormatContext* s);

  AVIOContext* (*avio_alloc_context)(
      unsigned char*,
      int,
      int,
      void*,
      int (*)(void*, uint8_t*, int),
      int (*)(void*, uint8_t*, int),
      int64_t (*)(void*, int64_t, int));

  const char* (*avio_enum_protocols)(void**, int);

  int (*avio_closep)(AVIOContext**);

  void (*avio_flush)(AVIOContext*);

  int (*avio_open2)(
      AVIOContext**,
      const char*,
      int,
      const AVIOInterruptCB*,
      AVDictionary**);

  AVFormatContext* (*avformat_alloc_context)();

  int (*avformat_alloc_output_context2)(
      AVFormatContext**,
      AVOutputFormat*,
      const char*,
      const char*);

  void (*avformat_close_input)(AVFormatContext**);

  int (*avformat_find_stream_info)(AVFormatContext*, AVDictionary**);

  void (*avformat_free_context)(AVFormatContext*);

  AVStream* (*avformat_new_stream)(AVFormatContext*, const AVCodec*);

  int (*avformat_open_input)(
      AVFormatContext**,
      const char*,
      AVFORMAT_CONST AVInputFormat*,
      AVDictionary**);

  unsigned (*avformat_version)();

  int (*avformat_write_header)(AVFormatContext*, AVDictionary**);

  /////////////////////////////////////////////////////////////////////////////
  // libavdevice
  /////////////////////////////////////////////////////////////////////////////

  void (*avdevice_register_all)();

  unsigned (*avdevice_version)();

  /////////////////////////////////////////////////////////////////////////////
  // libavfilter
  /////////////////////////////////////////////////////////////////////////////

  int (*av_buffersink_get_frame)(AVFilterContext*, AVFrame*);

  int (*av_buffersrc_add_frame_flags)(AVFilterContext*, AVFrame*, int);

  const AVFilter* (*avfilter_get_by_name)(const char*);

  AVFilterGraph* (*avfilter_graph_alloc)();

  int (*avfilter_graph_config)(AVFilterGraph*, void*);

  int (*avfilter_graph_create_filter)(
      AVFilterContext**,
      const AVFilter*,
      const char*,
      const char*,
      void*,
      AVFilterGraph*);

  void (*avfilter_graph_free)(AVFilterGraph**);

  int (*avfilter_graph_parse_ptr)(
      AVFilterGraph*,
      const char*,
      AVFilterInOut**,
      AVFilterInOut**,
      void*);

  AVFilterInOut* (*avfilter_inout_alloc)();

  void (*avfilter_inout_free)(AVFilterInOut**);

  unsigned (*avfilter_version)();
};

} // namespace torchaudio::io::detail

#endif
