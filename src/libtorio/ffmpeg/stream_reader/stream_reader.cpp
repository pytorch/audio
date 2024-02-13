#include <libtorio/ffmpeg/ffmpeg.h>
#include <libtorio/ffmpeg/stream_reader/stream_reader.h>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace torio::io {

using KeyType = StreamProcessor::KeyType;

//////////////////////////////////////////////////////////////////////////////
// Initialization / resource allocations
//////////////////////////////////////////////////////////////////////////////
namespace {
AVFormatContext* get_input_format_context(
    const std::string& src,
    const c10::optional<std::string>& format,
    const c10::optional<OptionDict>& option,
    AVIOContext* io_ctx) {
  AVFormatContext* p = avformat_alloc_context();
  TORCH_CHECK(p, "Failed to allocate AVFormatContext.");
  if (io_ctx) {
    p->pb = io_ctx;
  }

  auto* pInputFormat = [&format]() -> AVFORMAT_CONST AVInputFormat* {
    if (format.has_value()) {
      std::string format_str = format.value();
      AVFORMAT_CONST AVInputFormat* pInput =
          av_find_input_format(format_str.c_str());
      TORCH_CHECK(pInput, "Unsupported device/format: \"", format_str, "\"");
      return pInput;
    }
    return nullptr;
  }();

  AVDictionary* opt = get_option_dict(option);
  int ret = avformat_open_input(&p, src.c_str(), pInputFormat, &opt);
  clean_up_dict(opt);

  TORCH_CHECK(
      ret >= 0,
      "Failed to open the input \"",
      src,
      "\" (",
      av_err2string(ret),
      ").");
  return p;
}
} // namespace

StreamingMediaDecoder::StreamingMediaDecoder(AVFormatContext* p)
    : format_ctx(p) {
  C10_LOG_API_USAGE_ONCE("torchaudio.io.StreamingMediaDecoder");
  int ret = avformat_find_stream_info(format_ctx, nullptr);
  TORCH_CHECK(
      ret >= 0, "Failed to find stream information: ", av_err2string(ret));

  processors =
      std::vector<std::unique_ptr<StreamProcessor>>(format_ctx->nb_streams);
  for (int i = 0; i < format_ctx->nb_streams; ++i) {
    switch (format_ctx->streams[i]->codecpar->codec_type) {
      case AVMEDIA_TYPE_AUDIO:
      case AVMEDIA_TYPE_VIDEO:
        break;
      default:
        format_ctx->streams[i]->discard = AVDISCARD_ALL;
    }
  }
}

StreamingMediaDecoder::StreamingMediaDecoder(
    AVIOContext* io_ctx,
    const c10::optional<std::string>& format,
    const c10::optional<OptionDict>& option)
    : StreamingMediaDecoder(get_input_format_context(
          "Custom Input Context",
          format,
          option,
          io_ctx)) {}

StreamingMediaDecoder::StreamingMediaDecoder(
    const std::string& src,
    const c10::optional<std::string>& format,
    const c10::optional<OptionDict>& option)
    : StreamingMediaDecoder(
          get_input_format_context(src, format, option, nullptr)) {}

//////////////////////////////////////////////////////////////////////////////
// Helper methods
//////////////////////////////////////////////////////////////////////////////
void validate_open_stream(AVFormatContext* format_ctx) {
  TORCH_CHECK(format_ctx, "Stream is not open.");
}

void validate_src_stream_index(AVFormatContext* format_ctx, int i) {
  validate_open_stream(format_ctx);
  TORCH_CHECK(
      i >= 0 && i < static_cast<int>(format_ctx->nb_streams),
      "Source stream index out of range");
}

void validate_src_stream_type(
    AVFormatContext* format_ctx,
    int i,
    AVMediaType type) {
  validate_src_stream_index(format_ctx, i);
  TORCH_CHECK(
      format_ctx->streams[i]->codecpar->codec_type == type,
      "Stream ",
      i,
      " is not ",
      av_get_media_type_string(type),
      " stream.");
}

////////////////////////////////////////////////////////////////////////////////
// Query methods
////////////////////////////////////////////////////////////////////////////////
int64_t StreamingMediaDecoder::num_src_streams() const {
  return format_ctx->nb_streams;
}

namespace {
OptionDict parse_metadata(const AVDictionary* metadata) {
  AVDictionaryEntry* tag = nullptr;
  OptionDict ret;
  while ((tag = av_dict_get(metadata, "", tag, AV_DICT_IGNORE_SUFFIX))) {
    ret.emplace(std::string(tag->key), std::string(tag->value));
  }
  return ret;
}
} // namespace

OptionDict StreamingMediaDecoder::get_metadata() const {
  return parse_metadata(format_ctx->metadata);
}

SrcStreamInfo StreamingMediaDecoder::get_src_stream_info(int i) const {
  validate_src_stream_index(format_ctx, i);

  AVStream* stream = format_ctx->streams[i];
  AVCodecParameters* codecpar = stream->codecpar;

  SrcStreamInfo ret;
  ret.media_type = codecpar->codec_type;
  ret.bit_rate = codecpar->bit_rate;
  ret.num_frames = stream->nb_frames;
  ret.bits_per_sample = codecpar->bits_per_raw_sample;
  ret.metadata = parse_metadata(stream->metadata);
  const AVCodecDescriptor* desc = avcodec_descriptor_get(codecpar->codec_id);
  if (desc) {
    ret.codec_name = desc->name;
    ret.codec_long_name = desc->long_name;
  }

  switch (codecpar->codec_type) {
    case AVMEDIA_TYPE_AUDIO: {
      AVSampleFormat smp_fmt = static_cast<AVSampleFormat>(codecpar->format);
      if (smp_fmt != AV_SAMPLE_FMT_NONE) {
        ret.fmt_name = av_get_sample_fmt_name(smp_fmt);
      }
      ret.sample_rate = static_cast<double>(codecpar->sample_rate);
      ret.num_channels = codecpar->channels;
      break;
    }
    case AVMEDIA_TYPE_VIDEO: {
      AVPixelFormat pix_fmt = static_cast<AVPixelFormat>(codecpar->format);
      if (pix_fmt != AV_PIX_FMT_NONE) {
        ret.fmt_name = av_get_pix_fmt_name(pix_fmt);
      }
      ret.width = codecpar->width;
      ret.height = codecpar->height;
      ret.frame_rate = av_q2d(stream->r_frame_rate);
      break;
    }
    default:;
  }
  return ret;
}

namespace {
AVCodecParameters* get_codecpar() {
  AVCodecParameters* ptr = avcodec_parameters_alloc();
  TORCH_CHECK(ptr, "Failed to allocate resource.");
  return ptr;
}
} // namespace

StreamParams StreamingMediaDecoder::get_src_stream_params(int i) {
  validate_src_stream_index(format_ctx, i);
  AVStream* stream = format_ctx->streams[i];

  AVCodecParametersPtr codec_params(get_codecpar());
  int ret = avcodec_parameters_copy(codec_params, stream->codecpar);
  TORCH_CHECK(
      ret >= 0,
      "Failed to copy the stream's codec parameters. (",
      av_err2string(ret),
      ")");
  return {std::move(codec_params), stream->time_base, i};
}

int64_t StreamingMediaDecoder::num_out_streams() const {
  return static_cast<int64_t>(stream_indices.size());
}

OutputStreamInfo StreamingMediaDecoder::get_out_stream_info(int i) const {
  TORCH_CHECK(
      i >= 0 && static_cast<size_t>(i) < stream_indices.size(),
      "Output stream index out of range");
  int i_src = stream_indices[i].first;
  KeyType key = stream_indices[i].second;
  FilterGraphOutputInfo info = processors[i_src]->get_filter_output_info(key);

  OutputStreamInfo ret;
  ret.source_index = i_src;
  ret.filter_description = processors[i_src]->get_filter_description(key);
  ret.media_type = info.type;
  ret.format = info.format;
  switch (info.type) {
    case AVMEDIA_TYPE_AUDIO:
      ret.sample_rate = info.sample_rate;
      ret.num_channels = info.num_channels;
      break;
    case AVMEDIA_TYPE_VIDEO:
      ret.width = info.width;
      ret.height = info.height;
      ret.frame_rate = info.frame_rate;
      break;
    default:;
  }
  return ret;
}

int64_t StreamingMediaDecoder::find_best_audio_stream() const {
  return av_find_best_stream(
      format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
}

int64_t StreamingMediaDecoder::find_best_video_stream() const {
  return av_find_best_stream(
      format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
}

bool StreamingMediaDecoder::is_buffer_ready() const {
  if (processors.empty()) {
    // If no decoding output streams exist, then determine overall readiness
    // from the readiness of packet buffer.
    return packet_buffer->has_packets();
  } else {
    // Otherwise, determine readiness solely from the readiness of the decoding
    // output streams.
    for (const auto& it : processors) {
      if (it && !it->is_buffer_ready()) {
        return false;
      }
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Configure methods
////////////////////////////////////////////////////////////////////////////////
void StreamingMediaDecoder::seek(double timestamp_s, int64_t mode) {
  TORCH_CHECK(timestamp_s >= 0, "timestamp must be non-negative.");
  TORCH_CHECK(
      format_ctx->nb_streams > 0,
      "At least one stream must exist in this context");

  int64_t timestamp_av_tb = static_cast<int64_t>(timestamp_s * AV_TIME_BASE);

  int flag = AVSEEK_FLAG_BACKWARD;
  switch (mode) {
    case 0:
      // reset seek_timestap as it is only used for precise seek
      seek_timestamp = 0;
      break;
    case 1:
      flag |= AVSEEK_FLAG_ANY;
      // reset seek_timestap as it is only used for precise seek
      seek_timestamp = 0;
      break;
    case 2:
      seek_timestamp = timestamp_av_tb;
      break;
    default:
      TORCH_CHECK(false, "Invalid mode value: ", mode);
  }

  int ret = av_seek_frame(format_ctx, -1, timestamp_av_tb, flag);

  if (ret < 0) {
    seek_timestamp = 0;
    TORCH_CHECK(false, "Failed to seek. (" + av_err2string(ret) + ".)");
  }
  for (const auto& it : processors) {
    if (it) {
      it->flush();
      it->set_discard_timestamp(seek_timestamp);
    }
  }
}

void StreamingMediaDecoder::add_audio_stream(
    int64_t i,
    int64_t frames_per_chunk,
    int64_t num_chunks,
    const c10::optional<std::string>& filter_desc,
    const c10::optional<std::string>& decoder,
    const c10::optional<OptionDict>& decoder_option) {
  add_stream(
      static_cast<int>(i),
      AVMEDIA_TYPE_AUDIO,
      static_cast<int>(frames_per_chunk),
      static_cast<int>(num_chunks),
      filter_desc.value_or("anull"),
      decoder,
      decoder_option,
      torch::Device(torch::DeviceType::CPU));
}

void StreamingMediaDecoder::add_video_stream(
    int64_t i,
    int64_t frames_per_chunk,
    int64_t num_chunks,
    const c10::optional<std::string>& filter_desc,
    const c10::optional<std::string>& decoder,
    const c10::optional<OptionDict>& decoder_option,
    const c10::optional<std::string>& hw_accel) {
  const torch::Device device = [&]() {
    if (!hw_accel) {
      return torch::Device{c10::DeviceType::CPU};
    }
#ifdef USE_CUDA
    torch::Device d{hw_accel.value()};
    TORCH_CHECK(
        d.is_cuda(), "Only CUDA is supported for HW acceleration. Found: ", d);
    return d;
#else
    TORCH_CHECK(
        false,
        "torchaudio is not compiled with CUDA support. Hardware acceleration is not available.");
#endif
  }();

  add_stream(
      static_cast<int>(i),
      AVMEDIA_TYPE_VIDEO,
      static_cast<int>(frames_per_chunk),
      static_cast<int>(num_chunks),
      filter_desc.value_or("null"),
      decoder,
      decoder_option,
      device);
}

void StreamingMediaDecoder::add_packet_stream(int i) {
  validate_src_stream_index(format_ctx, i);
  if (!packet_buffer) {
    packet_buffer = std::make_unique<PacketBuffer>();
  }
  packet_stream_indices.emplace(i);
}

void StreamingMediaDecoder::add_stream(
    int i,
    AVMediaType media_type,
    int frames_per_chunk,
    int num_chunks,
    const std::string& filter_desc,
    const c10::optional<std::string>& decoder,
    const c10::optional<OptionDict>& decoder_option,
    const torch::Device& device) {
  validate_src_stream_type(format_ctx, i, media_type);

  AVStream* stream = format_ctx->streams[i];
  // When media source is file-like object, it is possible that source codec
  // is not detected properly.
  TORCH_CHECK(
      stream->codecpar->format != -1,
      "Failed to detect the source stream format.");

  if (!processors[i]) {
    processors[i] = std::make_unique<StreamProcessor>(stream->time_base);
    processors[i]->set_discard_timestamp(seek_timestamp);
  }
  if (!processors[i]->is_decoder_set()) {
    processors[i]->set_decoder(
        stream->codecpar, decoder, decoder_option, device);
  } else {
    TORCH_CHECK(
        !decoder && (!decoder_option || decoder_option.value().size() == 0),
        "Decoder options were provided, but the decoder has already been initialized.")
  }

  stream->discard = AVDISCARD_DEFAULT;

  auto frame_rate = [&]() -> AVRational {
    switch (media_type) {
      case AVMEDIA_TYPE_AUDIO:
        return AVRational{0, 1};
      case AVMEDIA_TYPE_VIDEO:
        return av_guess_frame_rate(format_ctx, stream, nullptr);
      default:
        TORCH_INTERNAL_ASSERT(
            false,
            "Unexpected media type is given: ",
            av_get_media_type_string(media_type));
    }
  }();
  int key = processors[i]->add_stream(
      frames_per_chunk, num_chunks, frame_rate, filter_desc, device);
  stream_indices.push_back(std::make_pair<>(i, key));
}

void StreamingMediaDecoder::remove_stream(int64_t i) {
  TORCH_CHECK(
      i >= 0 && static_cast<size_t>(i) < stream_indices.size(),
      "Output stream index out of range");
  auto it = stream_indices.begin() + i;
  int iP = it->first;
  processors[iP]->remove_stream(it->second);
  stream_indices.erase(it);

  // Check if the processor is still refered and if not, disable the processor
  bool still_used = false;
  for (auto& p : stream_indices) {
    still_used |= (iP == p.first);
    if (still_used) {
      break;
    }
  }
  if (!still_used) {
    processors[iP].reset(nullptr);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Stream methods
////////////////////////////////////////////////////////////////////////////////
// Note
// return value (to be finalized)
// 0: caller should keep calling this function
// 1: It's done, caller should stop calling
// <0: Some error happened
int StreamingMediaDecoder::process_packet() {
  int ret = av_read_frame(format_ctx, packet);
  if (ret == AVERROR_EOF) {
    ret = drain();
    return (ret < 0) ? ret : 1;
  }
  if (ret < 0) {
    return ret;
  }
  AutoPacketUnref auto_unref{packet};

  int stream_index = packet->stream_index;

  if (packet_stream_indices.count(stream_index)) {
    packet_buffer->push_packet(packet);
  }

  auto& processor = processors[stream_index];
  if (!processor) {
    return 0;
  }

  ret = processor->process_packet(packet);

  return (ret < 0) ? ret : 0;
}

// Similar to `process_packet()`, but in case process_packet returns EAGAIN,
// it keeps retrying until timeout happens,
//
// timeout and backoff is given in millisecond
int StreamingMediaDecoder::process_packet_block(
    double timeout,
    double backoff) {
  auto dead_line = [&]() {
    // If timeout < 0, then it repeats forever
    if (timeout < 0) {
      return std::chrono::time_point<std::chrono::steady_clock>::max();
    }
    auto timeout_ = static_cast<int64_t>(1000 * timeout);
    return std::chrono::steady_clock::now() +
        std::chrono::microseconds{timeout_};
  }();

  std::chrono::microseconds sleep{static_cast<int64_t>(1000 * backoff)};

  while (true) {
    int ret = process_packet();
    if (ret != AVERROR(EAGAIN)) {
      return ret;
    }
    if (dead_line < std::chrono::steady_clock::now()) {
      return ret;
    }
    // FYI: ffmpeg sleeps 10 milli seconds if the read happens in a separate
    // thread
    // https://github.com/FFmpeg/FFmpeg/blob/b0f8dbb0cacc45a19f18c043afc706d7d26bef74/fftools/ffmpeg.c#L3952
    // https://github.com/FFmpeg/FFmpeg/blob/b0f8dbb0cacc45a19f18c043afc706d7d26bef74/fftools/ffmpeg.c#L4542
    //
    std::this_thread::sleep_for(sleep);
  }
}

void StreamingMediaDecoder::process_all_packets() {
  int64_t ret = 0;
  do {
    ret = process_packet();
  } while (!ret);
}

int StreamingMediaDecoder::process_packet(
    const c10::optional<double>& timeout,
    const double backoff) {
  int code = [&]() -> int {
    if (timeout.has_value()) {
      return process_packet_block(timeout.value(), backoff);
    }
    return process_packet();
  }();
  TORCH_CHECK(
      code >= 0, "Failed to process a packet. (" + av_err2string(code) + "). ");
  return code;
}

int StreamingMediaDecoder::fill_buffer(
    const c10::optional<double>& timeout,
    const double backoff) {
  while (!is_buffer_ready()) {
    int code = process_packet(timeout, backoff);
    if (code != 0) {
      return code;
    }
  }
  return 0;
}

// <0: Some error happened.
int StreamingMediaDecoder::drain() {
  int ret = 0, tmp = 0;
  for (auto& p : processors) {
    if (p) {
      tmp = p->process_packet(nullptr);
      if (tmp < 0) {
        ret = tmp;
      }
    }
  }
  return ret;
}

std::vector<c10::optional<Chunk>> StreamingMediaDecoder::pop_chunks() {
  std::vector<c10::optional<Chunk>> ret;
  ret.reserve(static_cast<size_t>(num_out_streams()));
  for (auto& i : stream_indices) {
    ret.emplace_back(processors[i.first]->pop_chunk(i.second));
  }
  return ret;
}

std::vector<AVPacketPtr> StreamingMediaDecoder::pop_packets() {
  return packet_buffer->pop_packets();
}

//////////////////////////////////////////////////////////////////////////////
// StreamingMediaDecoderCustomIO
//////////////////////////////////////////////////////////////////////////////

namespace detail {
namespace {
AVIOContext* get_io_context(
    void* opaque,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence)) {
  unsigned char* buffer = static_cast<unsigned char*>(av_malloc(buffer_size));
  TORCH_CHECK(buffer, "Failed to allocate buffer.");
  AVIOContext* io_ctx = avio_alloc_context(
      buffer, buffer_size, 0, opaque, read_packet, nullptr, seek);
  if (!io_ctx) {
    av_freep(&buffer);
    TORCH_CHECK(false, "Failed to allocate AVIOContext.");
  }
  return io_ctx;
}
} // namespace

CustomInput::CustomInput(
    void* opaque,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence))
    : io_ctx(get_io_context(opaque, buffer_size, read_packet, seek)) {}
} // namespace detail

StreamingMediaDecoderCustomIO::StreamingMediaDecoderCustomIO(
    void* opaque,
    const c10::optional<std::string>& format,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence),
    const c10::optional<OptionDict>& option)
    : CustomInput(opaque, buffer_size, read_packet, seek),
      StreamingMediaDecoder(io_ctx, format, option) {}

namespace {
static int read_bytes(void* opaque, uint8_t* buf, int buf_size) {
  BytesWrapper* wrapper = static_cast<BytesWrapper*>(opaque);

  auto num_read = FFMIN(wrapper->src.size() - wrapper->index, buf_size);
  if (num_read == 0) {
    return AVERROR_EOF;
  }
  auto head = wrapper->src.data() + wrapper->index;
  memcpy(buf, head, num_read);
  wrapper->index += num_read;
  return num_read;
}

static int64_t seek_bytes(void* opaque, int64_t offset, int whence) {
  BytesWrapper* wrapper = static_cast<BytesWrapper*>(opaque);
  if (whence == AVSEEK_SIZE) {
    return wrapper->src.size();
  }

  if (whence == SEEK_SET) {
    wrapper->index = offset;
  } else if (whence == SEEK_CUR) {
    wrapper->index += offset;
  } else if (whence == SEEK_END) {
    wrapper->index = wrapper->src.size() + offset;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unexpected whence value: ", whence);
  }
  return static_cast<int64_t>(wrapper->index);
}
} // namespace

//////////////////////////////////////////////////////////////////////////////
// StreamingMediaDecoder Bytes
//////////////////////////////////////////////////////////////////////////////
StreamingMediaDecoderBytes::StreamingMediaDecoderBytes(
    std::string_view src,
    const c10::optional<std::string>& format,
    const c10::optional<std::map<std::string, std::string>>& option,
    int64_t buffer_size)
    : BytesWrapper{src},
      StreamingMediaDecoderCustomIO(
          this,
          format,
          buffer_size,
          read_bytes,
          seek_bytes,
          option) {}
} // namespace torio::io
