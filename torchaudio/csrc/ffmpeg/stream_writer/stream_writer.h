#pragma once

#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/filter_graph.h>

namespace torchaudio {
namespace ffmpeg {

struct OutputStream {
  AVStream* stream;
  AVCodecContextPtr codec_ctx;
  std::unique_ptr<FilterGraph> filter;
  AVFramePtr src_frame;
  AVFramePtr dst_frame;
  // The number of samples written so far
  int64_t num_frames;
  // Audio-only: The maximum frames that frame can hold
  int64_t frame_capacity;
  // Video-only: HW acceleration
  AVBufferRefPtr hw_device_ctx;
  AVBufferRefPtr hw_frame_ctx;
};

class StreamWriter {
  AVFormatOutputContextPtr pFormatContext;
  AVBufferRefPtr pHWBufferRef;
  std::vector<OutputStream> streams;
  AVPacketPtr pkt;

 public:
  explicit StreamWriter(AVFormatOutputContextPtr&& p);
  // Non-copyable
  StreamWriter(const StreamWriter&) = delete;
  StreamWriter& operator=(const StreamWriter&) = delete;

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  // Print the configured outputs
  void dump_format(int64_t i);

  //////////////////////////////////////////////////////////////////////////////
  // Configure methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  void add_audio_stream(
      int64_t sample_rate,
      int64_t num_channels,
      const std::string& format,
      const c10::optional<std::string>& encoder,
      const c10::optional<OptionDict>& encoder_option,
      const c10::optional<std::string>& encoder_format);
  void add_video_stream(
      double frame_rate,
      int64_t width,
      int64_t height,
      const std::string& format,
      const c10::optional<std::string>& encoder,
      const c10::optional<OptionDict>& encoder_option,
      const c10::optional<std::string>& encoder_format,
      const c10::optional<std::string>& hw_accel);
  void set_metadata(const OptionDict& metadata);

 private:
  AVStream* add_stream(AVCodecContextPtr& ctx);

  //////////////////////////////////////////////////////////////////////////////
  // Write methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  void open(const c10::optional<OptionDict>& opt);
  void close();

  void write_audio_chunk(int i, const torch::Tensor& chunk);
  void write_video_chunk(int i, const torch::Tensor& chunk);
  void flush();

 private:
  void validate_stream(int i, enum AVMediaType);
  void write_planar_video(
      OutputStream& os,
      const torch::Tensor& chunk,
      int num_planes);
  void write_interlaced_video(OutputStream& os, const torch::Tensor& chunk);
#ifdef USE_CUDA
  void write_planar_video_cuda(
      OutputStream& os,
      const torch::Tensor& chunk,
      int num_planes);
  void write_interlaced_video_cuda(
      OutputStream& os,
      const torch::Tensor& chunk,
      bool pad_extra = true);
#endif
  void process_frame(
      AVFrame* src_frame,
      std::unique_ptr<FilterGraph>& filter,
      AVFrame* dst_frame,
      AVCodecContextPtr& c,
      AVStream* st);
  void encode_frame(AVFrame* dst_frame, AVCodecContextPtr& c, AVStream* st);
  void flush_stream(OutputStream& os);
};

} // namespace ffmpeg
} // namespace torchaudio
