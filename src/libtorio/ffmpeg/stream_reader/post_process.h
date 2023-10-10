#pragma once
#include <libtorio/ffmpeg/filter_graph.h>
#include <libtorio/ffmpeg/stream_reader/typedefs.h>

namespace torchaudio::io {

struct IPostDecodeProcess {
  virtual ~IPostDecodeProcess() = default;

  virtual int process_frame(AVFrame* frame) = 0;
  virtual c10::optional<Chunk> pop_chunk() = 0;
  virtual bool is_buffer_ready() const = 0;
  virtual const std::string& get_filter_desc() const = 0;
  virtual FilterGraphOutputInfo get_filter_output_info() const = 0;
  virtual void flush() = 0;
};

std::unique_ptr<IPostDecodeProcess> get_audio_process(
    AVRational input_time_base,
    AVCodecContext* codec_ctx,
    const std::string& desc,
    int frames_per_chunk,
    int num_chunks);

std::unique_ptr<IPostDecodeProcess> get_video_process(
    AVRational input_time_base,
    AVRational frame_rate,
    AVCodecContext* codec_ctx,
    const std::string& desc,
    int frames_per_chunk,
    int num_chunks,
    const torch::Device& device);

} // namespace torchaudio::io
