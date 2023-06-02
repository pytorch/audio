#ifdef DLOPEN_FFMPEG

#include <ATen/DynamicLibrary.h>
#include <c10/util/CallOnce.h>
#include <torchaudio/csrc/ffmpeg/stub.h>

extern "C" {
#include <libavcodec/version.h>
#include <libavdevice/version.h>
#include <libavfilter/version.h>
#include <libavformat/version.h>
#include <libavutil/version.h>
}

namespace torchaudio::io::detail {
namespace {
class StubImpl {
  at::DynamicLibrary libavutil;
  at::DynamicLibrary libavcodec;
  at::DynamicLibrary libavformat;
  at::DynamicLibrary libavdevice;
  at::DynamicLibrary libavfilter;

 public:
  // The struct that holds all the function pointers to be used.
  FFmpegStub stub{};

  StubImpl(
      const char* util,
      const char* codec,
      const char* format,
      const char* device,
      const char* filter)
      : libavutil(util),
        libavcodec(codec),
        libavformat(format),
        libavdevice(device),
        libavfilter(filter) {
#define set(X) stub.X = (decltype(FFmpegStub::X))libavutil.sym(#X)
    set(av_buffer_ref);
    set(av_buffer_unref);
    set(av_d2q);
    set(av_dict_free);
    set(av_dict_get);
    set(av_dict_set);
    set(av_frame_alloc);
    set(av_frame_free);
    set(av_frame_get_buffer);
    set(av_frame_is_writable);
    set(av_frame_make_writable);
    set(av_frame_unref);
    set(av_freep);
    set(av_get_channel_layout_nb_channels);
    set(av_get_channel_name);
    set(av_get_default_channel_layout);
    set(av_get_media_type_string);
    set(av_get_pix_fmt);
    set(av_get_pix_fmt_name);
    set(av_get_sample_fmt);
    set(av_get_sample_fmt_name);
    set(av_get_time_base_q);
    set(av_hwdevice_ctx_create);
    set(av_hwframe_ctx_alloc);
    set(av_hwframe_ctx_init);
    set(av_hwframe_get_buffer);
    set(av_log_get_level);
    set(av_log_set_level);
    set(av_malloc);
    set(av_pix_fmt_desc_get);
    set(av_rescale_q);
    set(av_sample_fmt_is_planar);
    set(av_strdup);
    set(av_strerror);
    set(avutil_version);
#undef set

#define set(X) stub.X = (decltype(FFmpegStub::X))libavcodec.sym(#X)
    set(av_codec_is_decoder);
    set(av_codec_is_encoder);
    set(av_codec_iterate);
    set(av_packet_alloc);
    set(av_packet_clone);
    set(av_packet_free);
    set(av_packet_ref);
    set(av_packet_rescale_ts);
    set(av_packet_unref);
    set(avcodec_alloc_context3);
    set(avcodec_configuration);
    set(avcodec_descriptor_get);
    set(avcodec_find_decoder);
    set(avcodec_find_decoder_by_name);
    set(avcodec_find_encoder);
    set(avcodec_find_encoder_by_name);
    set(avcodec_flush_buffers);
    set(avcodec_free_context);
    set(avcodec_get_hw_config);
    set(avcodec_get_name);
    set(avcodec_open2);
    set(avcodec_parameters_alloc);
    set(avcodec_parameters_copy);
    set(avcodec_parameters_free);
    set(avcodec_parameters_from_context);
    set(avcodec_parameters_to_context);
    set(avcodec_receive_frame);
    set(avcodec_receive_packet);
    set(avcodec_send_frame);
    set(avcodec_send_packet);
    set(avcodec_version);
#undef set

#define set(X) stub.X = (decltype(FFmpegStub::X))libavformat.sym(#X)
    set(av_demuxer_iterate);
    set(av_dump_format);
    set(av_find_best_stream);
    set(av_find_input_format);
    set(av_guess_frame_rate);
    set(av_interleaved_write_frame);
    set(av_muxer_iterate);
    set(av_read_frame);
    set(av_seek_frame);
    set(av_write_trailer);
    set(avio_alloc_context);
    set(avio_enum_protocols);
    set(avio_closep);
    set(avio_flush);
    set(avio_open2);
    set(avformat_alloc_context);
    set(avformat_alloc_output_context2);
    set(avformat_close_input);
    set(avformat_find_stream_info);
    set(avformat_free_context);
    set(avformat_new_stream);
    set(avformat_open_input);
    set(avformat_version);
    set(avformat_write_header);
#undef set

#define set(X) stub.X = (decltype(FFmpegStub::X))libavdevice.sym(#X)
    set(avdevice_register_all);
    set(avdevice_version);
#undef set

#define set(X) stub.X = (decltype(FFmpegStub::X))libavfilter.sym(#X)
    set(av_buffersink_get_frame);
    set(av_buffersrc_add_frame_flags);
    set(avfilter_get_by_name);
    set(avfilter_graph_alloc);
    set(avfilter_graph_config);
    set(avfilter_graph_create_filter);
    set(avfilter_graph_free);
    set(avfilter_graph_parse_ptr);
    set(avfilter_inout_alloc);
    set(avfilter_inout_free);
    set(avfilter_version);
#undef set
  }
};

static std::unique_ptr<StubImpl> _stub;

void _init_stub() {
#if defined(_WIN32)
  _stub = std::make_unique<StubImpl>(
      "avutil-" AV_STRINGIFY(LIBAVUTIL_VERSION_MAJOR) ".dll",
      "avcodec-" AV_STRINGIFY(LIBAVCODEC_VERSION_MAJOR) ".dll",
      "avformat-" AV_STRINGIFY(LIBAVFORMAT_VERSION_MAJOR) ".dll",
      "avdevice-" AV_STRINGIFY(LIBAVDEVICE_VERSION_MAJOR) ".dll",
      "avfilter-" AV_STRINGIFY(LIBAVFILTER_VERSION_MAJOR) ".dll");
#elif defined(__APPLE__)
  _stub = std::make_unique<StubImpl>(
      "libavutil." AV_STRINGIFY(LIBAVUTIL_VERSION_MAJOR) ".dylib",
      "libavcodec." AV_STRINGIFY(LIBAVCODEC_VERSION_MAJOR) ".dylib",
      "libavformat." AV_STRINGIFY(LIBAVFORMAT_VERSION_MAJOR) ".dylib",
      "libavdevice." AV_STRINGIFY(LIBAVDEVICE_VERSION_MAJOR) ".dylib",
      "libavfilter." AV_STRINGIFY(LIBAVFILTER_VERSION_MAJOR) ".dylib");
#else
  _stub = std::make_unique<StubImpl>(
      "libavutil.so." AV_STRINGIFY(LIBAVUTIL_VERSION_MAJOR),
      "libavcodec.so." AV_STRINGIFY(LIBAVCODEC_VERSION_MAJOR),
      "libavformat.so." AV_STRINGIFY(LIBAVFORMAT_VERSION_MAJOR),
      "libavdevice.so." AV_STRINGIFY(LIBAVDEVICE_VERSION_MAJOR),
      "libavfilter.so." AV_STRINGIFY(LIBAVFILTER_VERSION_MAJOR));
#endif
}

} // namespace

FFmpegStub& ffmpeg_stub() {
  static c10::once_flag init_flag;
  c10::call_once(init_flag, _init_stub);
  return _stub->stub;
}

} // namespace torchaudio::io::detail

#endif
