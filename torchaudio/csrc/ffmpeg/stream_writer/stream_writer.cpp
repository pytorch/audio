#include <torchaudio/csrc/ffmpeg/stream_writer/stream_writer.h>
#include <torchaudio/csrc/ffmpeg/stub.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace torchaudio::io {
namespace {

AVFormatContext* get_output_format_context(
    const std::string& dst,
    const c10::optional<std::string>& format,
    AVIOContext* io_ctx) {
  if (io_ctx) {
    TORCH_CHECK(
        format,
        "`format` must be provided when the input is file-like object.");
  }

  AVFormatContext* p = nullptr;
  int ret = FFMPEG avformat_alloc_output_context2(
      &p, nullptr, format ? format.value().c_str() : nullptr, dst.c_str());
  TORCH_CHECK(
      ret >= 0,
      "Failed to open output \"",
      dst,
      "\" (",
      av_err2string(ret),
      ").");

  if (io_ctx) {
    p->pb = io_ctx;
    p->flags |= AVFMT_FLAG_CUSTOM_IO;
  }

  return p;
}
} // namespace

StreamWriter::StreamWriter(AVFormatContext* p) : format_ctx(p) {
  C10_LOG_API_USAGE_ONCE("torchaudio.io.StreamWriter");
}

StreamWriter::StreamWriter(
    AVIOContext* io_ctx,
    const c10::optional<std::string>& format)
    : StreamWriter(
          get_output_format_context("Custom Output Context", format, io_ctx)) {}

StreamWriter::StreamWriter(
    const std::string& dst,
    const c10::optional<std::string>& format)
    : StreamWriter(get_output_format_context(dst, format, nullptr)) {}

void StreamWriter::add_audio_stream(
    int sample_rate,
    int num_channels,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<int>& encoder_sample_rate,
    const c10::optional<int>& encoder_num_channels,
    const c10::optional<CodecConfig>& codec_config,
    const c10::optional<std::string>& filter_desc) {
  TORCH_CHECK(!is_open, "Output is already opened. Cannot add a new stream.");
  TORCH_INTERNAL_ASSERT(
      format_ctx->nb_streams == num_output_streams(),
      "The number of encode process and the number of output streams do not match.");
  processes.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(current_key),
      std::forward_as_tuple(get_audio_encode_process(
          format_ctx,
          sample_rate,
          num_channels,
          format,
          encoder,
          encoder_option,
          encoder_format,
          encoder_sample_rate,
          encoder_num_channels,
          codec_config,
          filter_desc)));
  current_key++;
}

void StreamWriter::add_video_stream(
    double frame_rate,
    int width,
    int height,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<double>& encoder_frame_rate,
    const c10::optional<int>& encoder_width,
    const c10::optional<int>& encoder_height,
    const c10::optional<std::string>& hw_accel,
    const c10::optional<CodecConfig>& codec_config,
    const c10::optional<std::string>& filter_desc) {
  TORCH_CHECK(!is_open, "Output is already opened. Cannot add a new stream.");
  TORCH_INTERNAL_ASSERT(
      format_ctx->nb_streams == num_output_streams(),
      "The number of encode process and the number of output streams do not match.");
  processes.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(current_key),
      std::forward_as_tuple(get_video_encode_process(
          format_ctx,
          frame_rate,
          width,
          height,
          format,
          encoder,
          encoder_option,
          encoder_format,
          encoder_frame_rate,
          encoder_width,
          encoder_height,
          hw_accel,
          codec_config,
          filter_desc)));
  current_key++;
}

void StreamWriter::add_packet_stream(const StreamParams& stream_params) {
  packet_writers.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(stream_params.stream_index),
      std::forward_as_tuple(format_ctx, stream_params));
  current_key++;
}

void StreamWriter::add_audio_frame_stream(
    int sample_rate,
    int num_channels,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<int>& encoder_sample_rate,
    const c10::optional<int>& encoder_num_channels,
    const c10::optional<CodecConfig>& codec_config,
    const c10::optional<std::string>& filter_desc) {
  TORCH_CHECK(!is_open, "Output is already opened. Cannot add a new stream.");
  TORCH_INTERNAL_ASSERT(
      format_ctx->nb_streams == num_output_streams(),
      "The number of encode process and the number of output streams do not match.");
  processes.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(current_key),
      std::forward_as_tuple(get_audio_encode_process(
          format_ctx,
          sample_rate,
          num_channels,
          format,
          encoder,
          encoder_option,
          encoder_format,
          encoder_sample_rate,
          encoder_num_channels,
          codec_config,
          filter_desc,
          true)));
  current_key++;
}

void StreamWriter::add_video_frame_stream(
    double frame_rate,
    int width,
    int height,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<double>& encoder_frame_rate,
    const c10::optional<int>& encoder_width,
    const c10::optional<int>& encoder_height,
    const c10::optional<std::string>& hw_accel,
    const c10::optional<CodecConfig>& codec_config,
    const c10::optional<std::string>& filter_desc) {
  TORCH_CHECK(!is_open, "Output is already opened. Cannot add a new stream.");
  TORCH_INTERNAL_ASSERT(
      format_ctx->nb_streams == num_output_streams(),
      "The number of encode process and the number of output streams do not match.");
  processes.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(current_key),
      std::forward_as_tuple(get_video_encode_process(
          format_ctx,
          frame_rate,
          width,
          height,
          format,
          encoder,
          encoder_option,
          encoder_format,
          encoder_frame_rate,
          encoder_width,
          encoder_height,
          hw_accel,
          codec_config,
          filter_desc,
          true)));
  current_key++;
}

void StreamWriter::set_metadata(const OptionDict& metadata) {
  FFMPEG av_dict_free(&format_ctx->metadata);
  for (auto const& [key, value] : metadata) {
    FFMPEG av_dict_set(&format_ctx->metadata, key.c_str(), value.c_str(), 0);
  }
}

void StreamWriter::dump_format(int64_t i) {
  FFMPEG av_dump_format(format_ctx, (int)i, format_ctx->url, 1);
}

void StreamWriter::open(const c10::optional<OptionDict>& option) {
  TORCH_INTERNAL_ASSERT(
      format_ctx->nb_streams == num_output_streams(),
      "The number of encode process and the number of output streams do not match.");

  int ret = 0;

  // Open the file if it was not provided by client code (i.e. when not
  // file-like object)
  AVFORMAT_CONST AVOutputFormat* fmt = format_ctx->oformat;
  AVDictionary* opt = get_option_dict(option);
  if (!(fmt->flags & AVFMT_NOFILE) &&
      !(format_ctx->flags & AVFMT_FLAG_CUSTOM_IO)) {
    ret = FFMPEG avio_open2(
        &format_ctx->pb, format_ctx->url, AVIO_FLAG_WRITE, nullptr, &opt);
    if (ret < 0) {
      FFMPEG av_dict_free(&opt);
      TORCH_CHECK(
          false,
          "Failed to open dst: ",
          format_ctx->url,
          " (",
          av_err2string(ret),
          ")");
    }
  }

  ret = FFMPEG avformat_write_header(format_ctx, &opt);
  clean_up_dict(opt);
  TORCH_CHECK(
      ret >= 0,
      "Failed to write header: ",
      format_ctx->url,
      " (",
      av_err2string(ret),
      ")");
  is_open = true;
}

void StreamWriter::close() {
  int ret = FFMPEG av_write_trailer(format_ctx);
  if (ret < 0) {
    LOG(WARNING) << "Failed to write trailer. (" << av_err2string(ret) << ").";
  }

  // Close the file if it was not provided by client code (i.e. when not
  // file-like object)
  AVFORMAT_CONST AVOutputFormat* fmt = format_ctx->oformat;
  if (!(fmt->flags & AVFMT_NOFILE) &&
      !(format_ctx->flags & AVFMT_FLAG_CUSTOM_IO)) {
    // avio_closep can be only applied to AVIOContext opened by avio_open
    FFMPEG avio_closep(&(format_ctx->pb));
  }
  is_open = false;
}

void StreamWriter::write_audio_chunk(
    int i,
    const torch::Tensor& waveform,
    const c10::optional<double>& pts) {
  TORCH_CHECK(is_open, "Output is not opened. Did you call `open` method?");
  TORCH_CHECK(
      0 <= i && i < static_cast<int>(format_ctx->nb_streams),
      "Invalid stream index. Index must be in range of [0, ",
      format_ctx->nb_streams,
      "). Found: ",
      i);
  TORCH_CHECK(
      format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO,
      "Stream ",
      i,
      " is not audio type.");
  processes.at(i).process(waveform, pts);
}

void StreamWriter::write_video_chunk(
    int i,
    const torch::Tensor& frames,
    const c10::optional<double>& pts) {
  TORCH_CHECK(is_open, "Output is not opened. Did you call `open` method?");
  TORCH_CHECK(
      0 <= i && i < static_cast<int>(format_ctx->nb_streams),
      "Invalid stream index. Index must be in range of [0, ",
      format_ctx->nb_streams,
      "). Found: ",
      i);
  TORCH_CHECK(
      format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO,
      "Stream ",
      i,
      " is not video type.");
  processes.at(i).process(frames, pts);
}

void StreamWriter::write_packet(const AVPacketPtr& packet) {
  TORCH_CHECK(is_open, "Output is not opened. Did you call `open` method?");
  int src_stream_index = packet->stream_index;
  TORCH_CHECK(
      packet_writers.count(src_stream_index),
      "Invalid packet stream source index ",
      src_stream_index);
  packet_writers.at(src_stream_index).write_packet(packet);
}

void StreamWriter::write_frame(int i, AVFrame* frame) {
  TORCH_CHECK(is_open, "Output is not opened. Did you call `open` method?");
  TORCH_CHECK(
      0 <= i && i < static_cast<int>(format_ctx->nb_streams),
      "Invalid stream index. Index must be in range of [0, ",
      format_ctx->nb_streams,
      "). Found: ",
      i);
  processes.at(i).process_frame(frame);
}

void StreamWriter::flush() {
  TORCH_CHECK(is_open, "Output is not opened. Did you call `open` method?");
  for (auto& p : processes) {
    p.second.flush();
  }
}

int StreamWriter::num_output_streams() {
  return static_cast<int>(processes.size() + packet_writers.size());
}

////////////////////////////////////////////////////////////////////////////////
// StreamWriterCustomIO
////////////////////////////////////////////////////////////////////////////////

namespace detail {
namespace {
AVIOContext* get_io_context(
    void* opaque,
    int buffer_size,
    int (*write_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence)) {
  unsigned char* buffer =
      static_cast<unsigned char*>(FFMPEG av_malloc(buffer_size));
  TORCH_CHECK(buffer, "Failed to allocate buffer.");
  AVIOContext* io_ctx = FFMPEG avio_alloc_context(
      buffer, buffer_size, 1, opaque, nullptr, write_packet, seek);
  if (!io_ctx) {
    FFMPEG av_freep(&buffer);
    TORCH_CHECK(false, "Failed to allocate AVIOContext.");
  }
  return io_ctx;
}
} // namespace

CustomOutput::CustomOutput(
    void* opaque,
    int buffer_size,
    int (*write_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence))
    : io_ctx(get_io_context(opaque, buffer_size, write_packet, seek)) {}
} // namespace detail

StreamWriterCustomIO::StreamWriterCustomIO(
    void* opaque,
    const c10::optional<std::string>& format,
    int buffer_size,
    int (*write_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence))
    : CustomOutput(opaque, buffer_size, write_packet, seek),
      StreamWriter(io_ctx, format) {}

} // namespace torchaudio::io
