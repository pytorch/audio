#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/stream_reader.h>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace torchaudio {
namespace io {

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

StreamReader::StreamReader(AVFormatContext* p) : pFormatContext(p) {
  int ret = avformat_find_stream_info(pFormatContext, nullptr);
  TORCH_CHECK(
      ret >= 0, "Failed to find stream information: ", av_err2string(ret));

  processors =
      std::vector<std::unique_ptr<StreamProcessor>>(pFormatContext->nb_streams);
  for (int i = 0; i < pFormatContext->nb_streams; ++i) {
    switch (pFormatContext->streams[i]->codecpar->codec_type) {
      case AVMEDIA_TYPE_AUDIO:
      case AVMEDIA_TYPE_VIDEO:
        break;
      default:
        pFormatContext->streams[i]->discard = AVDISCARD_ALL;
    }
  }
}

StreamReader::StreamReader(
    AVIOContext* io_ctx,
    const c10::optional<std::string>& format,
    const c10::optional<OptionDict>& option)
    : StreamReader(get_input_format_context(
          "Custom Input Context",
          format,
          option,
          io_ctx)) {}

StreamReader::StreamReader(
    const std::string& src,
    const c10::optional<std::string>& format,
    const c10::optional<OptionDict>& option)
    : StreamReader(get_input_format_context(src, format, option, nullptr)) {}

//////////////////////////////////////////////////////////////////////////////
// Helper methods
//////////////////////////////////////////////////////////////////////////////
void StreamReader::validate_open_stream() const {
  TORCH_CHECK(pFormatContext, "Stream is not open.");
}

void StreamReader::validate_src_stream_index(int i) const {
  validate_open_stream();
  TORCH_CHECK(
      i >= 0 && i < static_cast<int>(pFormatContext->nb_streams),
      "Source stream index out of range");
}

void StreamReader::validate_output_stream_index(int i) const {
  TORCH_CHECK(
      i >= 0 && i < static_cast<int>(stream_indices.size()),
      "Output stream index out of range");
}

void StreamReader::validate_src_stream_type(int i, AVMediaType type) {
  validate_src_stream_index(i);
  TORCH_CHECK(
      pFormatContext->streams[i]->codecpar->codec_type == type,
      "Stream ",
      i,
      " is not ",
      av_get_media_type_string(type),
      " stream.");
}

////////////////////////////////////////////////////////////////////////////////
// Query methods
////////////////////////////////////////////////////////////////////////////////
int64_t StreamReader::num_src_streams() const {
  return pFormatContext->nb_streams;
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

OptionDict StreamReader::get_metadata() const {
  return parse_metadata(pFormatContext->metadata);
}

SrcStreamInfo StreamReader::get_src_stream_info(int i) const {
  validate_src_stream_index(i);
  AVStream* stream = pFormatContext->streams[i];
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

int64_t StreamReader::num_out_streams() const {
  return static_cast<int64_t>(stream_indices.size());
}

OutputStreamInfo StreamReader::get_out_stream_info(int i) const {
  validate_output_stream_index(i);
  OutputStreamInfo ret;
  int i_src = stream_indices[i].first;
  KeyType key = stream_indices[i].second;
  ret.source_index = i_src;
  ret.filter_description = processors[i_src]->get_filter_description(key);
  return ret;
}

int64_t StreamReader::find_best_audio_stream() const {
  return av_find_best_stream(
      pFormatContext, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
}

int64_t StreamReader::find_best_video_stream() const {
  return av_find_best_stream(
      pFormatContext, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
}

bool StreamReader::is_buffer_ready() const {
  for (const auto& it : processors) {
    if (it && !it->is_buffer_ready()) {
      return false;
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Configure methods
////////////////////////////////////////////////////////////////////////////////
void StreamReader::seek(double timestamp_s, int64_t mode) {
  TORCH_CHECK(timestamp_s >= 0, "timestamp must be non-negative.");
  TORCH_CHECK(
      pFormatContext->nb_streams > 0,
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

  int ret = av_seek_frame(pFormatContext, -1, timestamp_av_tb, flag);

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

void StreamReader::add_audio_stream(
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
      filter_desc,
      decoder,
      decoder_option,
      torch::Device(torch::DeviceType::CPU));
}

void StreamReader::add_video_stream(
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
        d.type() == c10::DeviceType::CUDA,
        "Only CUDA is supported for hardware acceleration. Found: ",
        device.str());
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
      filter_desc,
      decoder,
      decoder_option,
      device);
}

void StreamReader::add_stream(
    int i,
    AVMediaType media_type,
    int frames_per_chunk,
    int num_chunks,
    const c10::optional<std::string>& filter_desc,
    const c10::optional<std::string>& decoder,
    const c10::optional<OptionDict>& decoder_option,
    const torch::Device& device) {
  validate_src_stream_type(i, media_type);

  AVStream* stream = pFormatContext->streams[i];
  // When media source is file-like object, it is possible that source codec is
  // not detected properly.
  TORCH_CHECK(
      stream->codecpar->format != -1,
      "Failed to detect the source stream format.");

  if (!processors[i]) {
    processors[i] = std::make_unique<StreamProcessor>(
        stream, decoder, decoder_option, device);
    processors[i]->set_discard_timestamp(seek_timestamp);
  }
  stream->discard = AVDISCARD_DEFAULT;
  int key = processors[i]->add_stream(
      frames_per_chunk, num_chunks, filter_desc, device);
  stream_indices.push_back(std::make_pair<>(i, key));
}

void StreamReader::remove_stream(int64_t i) {
  validate_output_stream_index(static_cast<int>(i));
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
int StreamReader::process_packet() {
  int ret = av_read_frame(pFormatContext, pPacket);
  if (ret == AVERROR_EOF) {
    ret = drain();
    return (ret < 0) ? ret : 1;
  }
  if (ret < 0) {
    return ret;
  }
  AutoPacketUnref packet{pPacket};
  auto& processor = processors[pPacket->stream_index];
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
int StreamReader::process_packet_block(double timeout, double backoff) {
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

// <0: Some error happened.
int StreamReader::drain() {
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

std::vector<c10::optional<Chunk>> StreamReader::pop_chunks() {
  std::vector<c10::optional<Chunk>> ret;
  ret.reserve(static_cast<size_t>(num_out_streams()));
  for (auto& i : stream_indices) {
    ret.emplace_back(processors[i.first]->pop_chunk(i.second));
  }
  return ret;
}

} // namespace io
} // namespace torchaudio
