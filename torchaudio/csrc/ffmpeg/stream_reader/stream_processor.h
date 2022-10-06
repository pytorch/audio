#pragma once

#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/decoder.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/sink.h>
#include <map>

namespace torchaudio {
namespace ffmpeg {

class StreamProcessor {
 public:
  using KeyType = int;

 private:
  AVFramePtr pFrame1;
  AVFramePtr pFrame2;

  // Components for decoding source media
  double decoder_time_base; // for debug
  Decoder decoder;

  KeyType current_key = 0;
  std::map<KeyType, Sink> sinks;

 public:
  StreamProcessor(
      AVCodecParameters* codecpar,
      const c10::optional<std::string>& decoder_name,
      const c10::optional<OptionDict>& decoder_option,
      const torch::Device& device);
  ~StreamProcessor() = default;
  // Non-copyable
  StreamProcessor(const StreamProcessor&) = delete;
  StreamProcessor& operator=(const StreamProcessor&) = delete;
  // Movable
  StreamProcessor(StreamProcessor&&) = default;
  StreamProcessor& operator=(StreamProcessor&&) = default;

  //////////////////////////////////////////////////////////////////////////////
  // Configurations
  //////////////////////////////////////////////////////////////////////////////
  // 1. Initialize decoder (if not initialized yet)
  // 2. Configure a new audio/video filter.
  //    If the custom parameter is provided, then perform resize, resample etc..
  //    otherwise, the filter only converts the sample type.
  // 3. Configure a buffer.
  // 4. Return filter ID.
  KeyType add_stream(
      AVRational input_time_base,
      AVCodecParameters* codecpar,
      int frames_per_chunk,
      int num_chunks,
      const c10::optional<std::string>& filter_description,
      const torch::Device& device);

  // 1. Remove the stream
  void remove_stream(KeyType key);

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
  std::string get_filter_description(KeyType key) const;
  bool is_buffer_ready() const;

  //////////////////////////////////////////////////////////////////////////////
  // The streaming process
  //////////////////////////////////////////////////////////////////////////////
  // 1. decode the input frame
  // 2. pass the decoded data to filters
  // 3. each filter store the result to the corresponding buffer
  // - Sending NULL will drain (flush) the internal
  int process_packet(AVPacket* packet, int64_t discard_before_pts = -1);

  // flush the internal buffer of decoder.
  // To be use when seeking
  void flush();

 private:
  int send_frame(AVFrame* pFrame);

  //////////////////////////////////////////////////////////////////////////////
  // Retrieval
  //////////////////////////////////////////////////////////////////////////////
 public:
  // Get the chunk from the given filter result
  c10::optional<torch::Tensor> pop_chunk(KeyType key);
};

} // namespace ffmpeg
} // namespace torchaudio
