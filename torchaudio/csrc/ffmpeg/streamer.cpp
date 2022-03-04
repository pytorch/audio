#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/streamer.h>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace torchaudio {
namespace ffmpeg {

using KeyType = StreamProcessor::KeyType;

//////////////////////////////////////////////////////////////////////////////
// Helper methods
//////////////////////////////////////////////////////////////////////////////
void Streamer::validate_open_stream() const {
  if (!pFormatContext)
    throw std::runtime_error("Stream is not open.");
}

void Streamer::validate_src_stream_index(int i) const {
  validate_open_stream();
  if (i < 0 || i >= static_cast<int>(pFormatContext->nb_streams))
    throw std::out_of_range("Source stream index out of range");
}

void Streamer::validate_output_stream_index(int i) const {
  if (i < 0 || i >= static_cast<int>(stream_indices.size()))
    throw std::out_of_range("Output stream index out of range");
}

void Streamer::validate_src_stream_type(int i, AVMediaType type) {
  validate_src_stream_index(i);
  if (pFormatContext->streams[i]->codecpar->codec_type != type) {
    std::ostringstream oss;
    oss << "Stream " << i << " is not " << av_get_media_type_string(type)
        << " stream.";
    throw std::runtime_error(oss.str());
  }
}

//////////////////////////////////////////////////////////////////////////////
// Initialization / resource allocations
//////////////////////////////////////////////////////////////////////////////
Streamer::Streamer(
    const std::string& src,
    const std::string& device,
    const std::map<std::string, std::string>& option)
    : pFormatContext(src, device, option) {
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

////////////////////////////////////////////////////////////////////////////////
// Query methods
////////////////////////////////////////////////////////////////////////////////
int Streamer::num_src_streams() const {
  return pFormatContext->nb_streams;
}

SrcStreamInfo Streamer::get_src_stream_info(int i) const {
  validate_src_stream_index(i);
  AVStream* stream = pFormatContext->streams[i];
  AVCodecParameters* codecpar = stream->codecpar;

  SrcStreamInfo ret;
  ret.media_type = codecpar->codec_type;
  ret.bit_rate = codecpar->bit_rate;
  const AVCodecDescriptor* desc = avcodec_descriptor_get(codecpar->codec_id);
  if (desc) {
    ret.codec_name = desc->name;
    ret.codec_long_name = desc->long_name;
  }
  switch (codecpar->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
      ret.fmt_name =
          av_get_sample_fmt_name(static_cast<AVSampleFormat>(codecpar->format));
      ret.sample_rate = static_cast<double>(codecpar->sample_rate);
      ret.num_channels = codecpar->channels;
      break;
    case AVMEDIA_TYPE_VIDEO:
      ret.fmt_name =
          av_get_pix_fmt_name(static_cast<AVPixelFormat>(codecpar->format));
      ret.width = codecpar->width;
      ret.height = codecpar->height;
      ret.frame_rate = av_q2d(stream->r_frame_rate);
      break;
    default:;
  }
  return ret;
}

int Streamer::num_out_streams() const {
  return stream_indices.size();
}

OutputStreamInfo Streamer::get_out_stream_info(int i) const {
  validate_output_stream_index(i);
  OutputStreamInfo ret;
  int i_src = stream_indices[i].first;
  KeyType key = stream_indices[i].second;
  ret.source_index = i_src;
  ret.filter_description = processors[i_src]->get_filter_description(key);
  return ret;
}

int Streamer::find_best_audio_stream() const {
  return av_find_best_stream(
      pFormatContext, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
}

int Streamer::find_best_video_stream() const {
  return av_find_best_stream(
      pFormatContext, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
}

bool Streamer::is_buffer_ready() const {
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
void Streamer::seek(double timestamp) {
  if (timestamp < 0) {
    throw std::invalid_argument("timestamp must be non-negative.");
  }

  int64_t ts = static_cast<int64_t>(timestamp * AV_TIME_BASE);
  int ret = avformat_seek_file(pFormatContext, -1, INT64_MIN, ts, INT64_MAX, 0);
  if (ret < 0) {
    throw std::runtime_error("Failed to seek. (" + av_err2string(ret) + ".)");
  }
  for (const auto& it : processors) {
    if (it) {
      it->flush();
    }
  }
}

void Streamer::add_audio_stream(
    int i,
    int frames_per_chunk,
    int num_chunks,
    std::string filter_desc) {
  add_stream(
      i,
      AVMEDIA_TYPE_AUDIO,
      frames_per_chunk,
      num_chunks,
      std::move(filter_desc));
}

void Streamer::add_video_stream(
    int i,
    int frames_per_chunk,
    int num_chunks,
    std::string filter_desc) {
  add_stream(
      i,
      AVMEDIA_TYPE_VIDEO,
      frames_per_chunk,
      num_chunks,
      std::move(filter_desc));
}

void Streamer::add_stream(
    int i,
    AVMediaType media_type,
    int frames_per_chunk,
    int num_chunks,
    std::string filter_desc) {
  validate_src_stream_type(i, media_type);
  AVStream* stream = pFormatContext->streams[i];
  stream->discard = AVDISCARD_DEFAULT;
  if (!processors[i])
    processors[i] = std::make_unique<StreamProcessor>(stream->codecpar);
  int key = processors[i]->add_stream(
      stream->time_base,
      stream->codecpar,
      frames_per_chunk,
      num_chunks,
      std::move(filter_desc));
  stream_indices.push_back(std::make_pair<>(i, key));
}

void Streamer::remove_stream(int i) {
  validate_output_stream_index(i);
  auto it = stream_indices.begin() + i;
  int iP = it->first;
  processors[iP]->remove_stream(it->second);
  stream_indices.erase(it);

  // Check if the processor is still refered and if not, disable the processor
  bool still_used = false;
  for (auto& p : stream_indices) {
    still_used |= (iP == p.first);
    if (still_used)
      break;
  }
  if (!still_used)
    processors[iP].reset(NULL);
}

////////////////////////////////////////////////////////////////////////////////
// Stream methods
////////////////////////////////////////////////////////////////////////////////
// Note
// return value (to be finalized)
// 0: caller should keep calling this function
// 1: It's done, caller should stop calling
// <0: Some error happened
int Streamer::process_packet() {
  int ret = av_read_frame(pFormatContext, pPacket);
  if (ret == AVERROR_EOF) {
    ret = drain();
    return (ret < 0) ? ret : 1;
  }
  if (ret < 0)
    return ret;
  AutoPacketUnref packet{pPacket};
  auto& processor = processors[pPacket->stream_index];
  if (!processor)
    return 0;
  ret = processor->process_packet(packet);
  return (ret < 0) ? ret : 0;
}

// Similar to `process_packet()`, but in case process_packet returns EAGAIN,
// it keeps retrying until timeout happens,
//
// timeout and backoff is given in millisecond
int Streamer::process_packet_block(double timeout, double backoff) {
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
    // ffmpeg sleeps 10 milli seconds if the read happens in a separate thread
    // https://github.com/FFmpeg/FFmpeg/blob/b0f8dbb0cacc45a19f18c043afc706d7d26bef74/fftools/ffmpeg.c#L3952
    // https://github.com/FFmpeg/FFmpeg/blob/b0f8dbb0cacc45a19f18c043afc706d7d26bef74/fftools/ffmpeg.c#L4542
    //
    // But it does not seem to sleep when running in single thread.
    // Empirically we observed that the streaming result is worse with sleep.
    // busy-waiting is not a recommended way to resolve this, but after simple
    // testing, there wasn't a noticible difference in CPU utility. So we do not
    // sleep here.
    //
    std::this_thread::sleep_for(sleep);
  }
}

// <0: Some error happened.
int Streamer::drain() {
  int ret = 0, tmp = 0;
  for (auto& p : processors) {
    if (p) {
      tmp = p->process_packet(NULL);
      if (tmp < 0)
        ret = tmp;
    }
  }
  return ret;
}

std::vector<c10::optional<torch::Tensor>> Streamer::pop_chunks() {
  std::vector<c10::optional<torch::Tensor>> ret;
  for (auto& i : stream_indices) {
    ret.push_back(processors[i.first]->pop_chunk(i.second));
  }
  return ret;
}

} // namespace ffmpeg
} // namespace torchaudio
