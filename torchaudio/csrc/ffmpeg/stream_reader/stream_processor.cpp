#include <torchaudio/csrc/ffmpeg/stream_reader/stream_processor.h>
#include <stdexcept>

namespace torchaudio {
namespace ffmpeg {

using KeyType = StreamProcessor::KeyType;

StreamProcessor::StreamProcessor(
    AVCodecParameters* codecpar,
    const c10::optional<std::string>& decoder_name,
    const c10::optional<OptionDict>& decoder_option,
    const torch::Device& device)
    : decoder(codecpar, decoder_name, decoder_option, device) {}

////////////////////////////////////////////////////////////////////////////////
// Configurations
////////////////////////////////////////////////////////////////////////////////
KeyType StreamProcessor::add_stream(
    AVRational input_time_base,
    AVCodecParameters* codecpar,
    int frames_per_chunk,
    int num_chunks,
    const c10::optional<std::string>& filter_description,
    const torch::Device& device) {
  switch (codecpar->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
    case AVMEDIA_TYPE_VIDEO:
      break;
    default:
      TORCH_CHECK(false, "Only Audio and Video are supported");
  }
  KeyType key = current_key++;
  sinks.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(key),
      std::forward_as_tuple(
          input_time_base,
          codecpar,
          frames_per_chunk,
          num_chunks,
          filter_description,
          device));
  decoder_time_base = av_q2d(input_time_base);
  return key;
}

void StreamProcessor::remove_stream(KeyType key) {
  sinks.erase(key);
}

////////////////////////////////////////////////////////////////////////////////
// Query methods
////////////////////////////////////////////////////////////////////////////////
std::string StreamProcessor::get_filter_description(KeyType key) const {
  return sinks.at(key).get_filter_description();
}

bool StreamProcessor::is_buffer_ready() const {
  for (const auto& it : sinks) {
    if (!it.second.is_buffer_ready()) {
      return false;
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// The streaming process
////////////////////////////////////////////////////////////////////////////////
// 0: some kind of success
// <0: Some error happened
int StreamProcessor::process_packet(
    AVPacket* packet,
    int64_t discard_before_pts) {
  int ret = decoder.process_packet(packet);
  while (ret >= 0) {
    ret = decoder.get_frame(pFrame1);
    //  AVERROR(EAGAIN) means that new input data is required to return new
    //  output.
    if (ret == AVERROR(EAGAIN))
      return 0;
    if (ret == AVERROR_EOF)
      return send_frame(NULL);
    if (ret < 0)
      return ret;

    if (pFrame1->pts >= discard_before_pts)
      send_frame(pFrame1);

    // else we can just unref the frame and continue
    av_frame_unref(pFrame1);
  }
  return ret;
}

void StreamProcessor::flush() {
  decoder.flush_buffer();
  for (auto& ite : sinks) {
    ite.second.flush();
  }
}

// 0: some kind of success
// <0: Some error happened
int StreamProcessor::send_frame(AVFrame* pFrame) {
  int ret = 0;
  for (auto& ite : sinks) {
    int ret2 = ite.second.process_frame(pFrame);
    if (ret2 < 0)
      ret = ret2;
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Retrieval
////////////////////////////////////////////////////////////////////////////////
c10::optional<torch::Tensor> StreamProcessor::pop_chunk(KeyType key) {
  return sinks.at(key).buffer->pop_chunk();
}

} // namespace ffmpeg
} // namespace torchaudio
