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

///
/// Encode and write audio/video streams chunk by chunk
///
class StreamWriter {
  AVFormatOutputContextPtr pFormatContext;
  AVBufferRefPtr pHWBufferRef;
  std::vector<OutputStream> streams;
  AVPacketPtr pkt;

 protected:
  explicit StreamWriter(AVFormatContext*);

 public:
  /// Construct StreamWriter from destination URI
  ///
  /// @param dst Destination where encoded data are written.
  /// @param format Specify output format. If not provided, it is guessed from
  /// ``dst``.
  explicit StreamWriter(
      const std::string& dst,
      const c10::optional<std::string>& format = {});

  /// Construct StreamWriter from custom IO
  ///
  /// @param io_ctx Custom IO.
  /// @param format Specify output format.
  // TODO: Move this into wrapper class.
  explicit StreamWriter(
      AVIOContext* io_ctx,
      const c10::optional<std::string>& format);

  // Non-copyable
  StreamWriter(const StreamWriter&) = delete;
  StreamWriter& operator=(const StreamWriter&) = delete;

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  /// @internal

  /// Print the configured outputs
  void dump_format(int64_t i);

  /// @endinternal

  //////////////////////////////////////////////////////////////////////////////
  // Configure methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  /// Add an output audio stream.
  ///
  /// @param sample_rate The sample rate.
  /// @param num_channels The number of channels.
  /// @param format Input sample format, which determines the dtype
  /// of the input tensor.
  /// @parblock
  ///
  /// - ``"u8"``: The input tensor must be ``torch.uint8`` type.
  /// - ``"s16"``: The input tensor must be ``torch.int16`` type.
  /// - ``"s32"``: The input tensor must be ``torch.int32`` type.
  /// - ``"s64"``: The input tensor must be ``torch.int64`` type.
  /// - ``"flt"``: The input tensor must be ``torch.float32`` type.
  /// - ``"dbl"``: The input tensor must be ``torch.float64`` type.
  ///
  /// Default: ``"flt"``.
  /// @endparblock
  /// @param encoder The name of the encoder to be used.
  /// @parblock
  /// When provided, use the specified encoder instead of the default one.
  ///
  /// To list the available encoders, you can use ``ffmpeg -encoders`` command.
  /// @endparblock
  /// @param encoder_option Options passed to encoder.
  /// To list encoder options for a encoder, you can use
  /// ``ffmpeg -h encoder=<ENCODER>``.
  /// @param encoder_format Format used to encode media.
  /// When encoder supports multiple formats, passing this argument will
  /// override the format used for encoding.
  ///  To list supported formats for the encoder, you can use
  /// ``ffmpeg -h encoder=<ENCODER>`` command.
  void add_audio_stream(
      int64_t sample_rate,
      int64_t num_channels,
      const std::string& format,
      const c10::optional<std::string>& encoder,
      const c10::optional<OptionDict>& encoder_option,
      const c10::optional<std::string>& encoder_format);
  /// Add an output video stream.
  ///
  /// @param frame_rate Frame rate
  /// @param width Width
  /// @param height Height
  /// @param format Input pixel format, which determines the
  /// color channel order of the input tensor.
  /// @parblock
  ///
  /// - ``"gray8"``: One channel, grayscale.
  /// - ``"rgb24"``: Three channels in the order of RGB.
  /// - ``"bgr24"``: Three channels in the order of BGR.
  /// - ``"yuv444p"``: Three channels in the order of YUV.
  ///
  /// In either case, the input tensor has to be ``torch.uint8`` type and
  /// the shape must be (frame, channel, height, width).
  /// @endparblock
  /// @param encoder See ``add_audio_stream()``.
  /// @param encoder_option See ``add_audio_stream()``.
  /// @param encoder_format See ``add_audio_stream()``.
  /// @param hw_accel Enable hardware acceleration.
  /// @parblock
  /// When video is encoded on CUDA hardware, for example
  /// `encoder="h264_nvenc"`, passing CUDA device indicator to `hw_accel`
  /// (i.e. `hw_accel="cuda:0"`) will make StreamWriter expect video
  /// chunk to be a CUDA Tensor. Passing CPU Tensor will result in an error.
  ///
  /// If `None`, the video chunk Tensor has to be a CPU Tensor.
  /// @endparblock
  void add_video_stream(
      double frame_rate,
      int64_t width,
      int64_t height,
      const std::string& format,
      const c10::optional<std::string>& encoder,
      const c10::optional<OptionDict>& encoder_option,
      const c10::optional<std::string>& encoder_format,
      const c10::optional<std::string>& hw_accel);
  /// Set file-level metadata
  /// @param metadata metadata.
  void set_metadata(const OptionDict& metadata);

 private:
  AVStream* add_stream(AVCodecContextPtr& ctx);

  //////////////////////////////////////////////////////////////////////////////
  // Write methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  /// Open the output file / device and write the header.
  ///
  /// @param opt Private options for protocol, device and muxer.
  void open(const c10::optional<OptionDict>& opt);
  /// Close the output file / device and finalize metadata.
  void close();

  /// Write audio data
  /// @param i Stream index.
  /// @param chunk Waveform tensor. Shape: ``(frame, channel)``.
  /// The ``dtype`` must match what was passed to ``add_audio_stream()`` method.
  void write_audio_chunk(int i, const torch::Tensor& chunk);
  /// Write video data
  /// @param i Stream index.
  /// @param chunk Video/image tensor. Shape: ``(time, channel, height,
  /// width)``. The ``dtype`` must be ``torch.uint8``. The shape ``(height,
  /// width and the number of channels)`` must match what was configured when
  /// calling ``add_video_stream()``.
  void write_video_chunk(int i, const torch::Tensor& chunk);
  /// Flush the frames from encoders and write the frames to the destination.
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
