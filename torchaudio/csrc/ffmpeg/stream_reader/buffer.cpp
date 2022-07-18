#include <torchaudio/csrc/ffmpeg/stream_reader/buffer.h>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace torchaudio {
namespace ffmpeg {

Buffer::Buffer(int frames_per_chunk, int num_chunks)
    : frames_per_chunk(frames_per_chunk), num_chunks(num_chunks) {}

AudioBuffer::AudioBuffer(int frames_per_chunk, int num_chunks)
    : Buffer(frames_per_chunk, num_chunks) {}

VideoBuffer::VideoBuffer(
    int frames_per_chunk,
    int num_chunks,
    const torch::Device& device_)
    : Buffer(frames_per_chunk, num_chunks), device(device_) {}

////////////////////////////////////////////////////////////////////////////////
// Query
////////////////////////////////////////////////////////////////////////////////
bool Buffer::is_ready() const {
  if (frames_per_chunk < 0)
    return num_buffered_frames > 0;
  return num_buffered_frames >= frames_per_chunk;
}

////////////////////////////////////////////////////////////////////////////////
// Modifiers - Push Audio
////////////////////////////////////////////////////////////////////////////////
namespace {
torch::Tensor convert_audio_tensor(AVFrame* pFrame) {
  // ref: https://ffmpeg.org/doxygen/4.1/filter__audio_8c_source.html#l00215
  AVSampleFormat format = static_cast<AVSampleFormat>(pFrame->format);
  int num_channels = pFrame->channels;
  int bps = av_get_bytes_per_sample(format);

  // Note
  // FFMpeg's `nb_samples` represnts the number of samples par channel.
  // This corresponds to `num_frames` in torchaudio's notation.
  // Also torchaudio uses `num_samples` as the number of samples
  // across channels.
  int num_frames = pFrame->nb_samples;

  int is_planar = av_sample_fmt_is_planar(format);
  int num_planes = is_planar ? num_channels : 1;
  int plane_size = bps * num_frames * (is_planar ? 1 : num_channels);
  std::vector<int64_t> shape = is_planar
      ? std::vector<int64_t>{num_channels, num_frames}
      : std::vector<int64_t>{num_frames, num_channels};

  torch::Tensor t;
  uint8_t* ptr = NULL;
  switch (format) {
    case AV_SAMPLE_FMT_U8:
    case AV_SAMPLE_FMT_U8P: {
      t = torch::empty(shape, torch::kUInt8);
      ptr = t.data_ptr<uint8_t>();
      break;
    }
    case AV_SAMPLE_FMT_S16:
    case AV_SAMPLE_FMT_S16P: {
      t = torch::empty(shape, torch::kInt16);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<int16_t>());
      break;
    }
    case AV_SAMPLE_FMT_S32:
    case AV_SAMPLE_FMT_S32P: {
      t = torch::empty(shape, torch::kInt32);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<int32_t>());
      break;
    }
    case AV_SAMPLE_FMT_S64:
    case AV_SAMPLE_FMT_S64P: {
      t = torch::empty(shape, torch::kInt64);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<int64_t>());
      break;
    }
    case AV_SAMPLE_FMT_FLT:
    case AV_SAMPLE_FMT_FLTP: {
      t = torch::empty(shape, torch::kFloat32);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<float>());
      break;
    }
    case AV_SAMPLE_FMT_DBL:
    case AV_SAMPLE_FMT_DBLP: {
      t = torch::empty(shape, torch::kFloat64);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<double>());
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported audio format: " +
              std::string(av_get_sample_fmt_name(format)));
  }
  for (int i = 0; i < num_planes; ++i) {
    memcpy(ptr, pFrame->extended_data[i], plane_size);
    ptr += plane_size;
  }
  if (is_planar)
    t = t.t();
  return t;
}
} // namespace

void AudioBuffer::push_tensor(torch::Tensor t) {
  // If frames_per_chunk < 0, users want to fetch all frames.
  // Just push back to chunks and that's it.
  if (frames_per_chunk < 0) {
    chunks.push_back(t);
    num_buffered_frames += t.size(0);
    return;
  }

  // Push
  // Note:
  // For audio, the incoming tensor contains multiple of samples.
  // For small `frames_per_chunk` value, it might be more than `max_frames`.
  // If we push the tensor as-is, then, the whole frame might be popped at
  // trimming stage, resulting buffer always empty. So we slice push the
  // incoming Tensor.

  // Check the last inserted Tensor and if the numbe of frames is not
  // frame_per_chunk, reprocess it again with the incomping tensor
  if (num_buffered_frames % frames_per_chunk) {
    torch::Tensor prev = chunks.back();
    chunks.pop_back();
    num_buffered_frames -= prev.size(0);
    t = torch::cat({prev, t}, 0);
  }

  while (true) {
    int num_input_frames = t.size(0);
    if (num_input_frames <= frames_per_chunk) {
      chunks.push_back(t);
      num_buffered_frames += num_input_frames;
      break;
    }
    // The input tensor contains more frames than frames_per_chunk
    auto splits = torch::tensor_split(t, {frames_per_chunk, num_input_frames});
    chunks.push_back(splits[0]);
    num_buffered_frames += frames_per_chunk;
    t = splits[1];
  }

  // Trim
  // If frames_per_chunk > 0, we only retain the following number of frames and
  // Discard older frames.
  int max_frames = num_chunks * frames_per_chunk;
  while (num_buffered_frames > max_frames) {
    TORCH_WARN_ONCE(
        "The number of buffered frames exceeded the buffer size. "
        "Dropping the old frames. "
        "To avoid this, you can set a higher buffer_chunk_size value.");
    torch::Tensor& t = chunks.front();
    num_buffered_frames -= t.size(0);
    chunks.pop_front();
  }
}

void AudioBuffer::push_frame(AVFrame* frame) {
  push_tensor(convert_audio_tensor(frame));
}

////////////////////////////////////////////////////////////////////////////////
// Modifiers - Push Video
////////////////////////////////////////////////////////////////////////////////
namespace {
torch::Tensor convert_interlaced_video(AVFrame* pFrame) {
  int width = pFrame->width;
  int height = pFrame->height;
  uint8_t* buf = pFrame->data[0];
  int linesize = pFrame->linesize[0];
  int channel = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(pFrame->format))
                    ->nb_components;

  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .layout(torch::kStrided)
                     .device(torch::kCPU);

  torch::Tensor frame = torch::empty({1, height, width, channel}, options);
  auto ptr = frame.data_ptr<uint8_t>();
  int stride = width * channel;
  for (int i = 0; i < height; ++i) {
    memcpy(ptr, buf, stride);
    buf += linesize;
    ptr += stride;
  }
  return frame.permute({0, 3, 1, 2});
}

torch::Tensor convert_planar_video(AVFrame* pFrame) {
  int width = pFrame->width;
  int height = pFrame->height;
  int num_planes =
      av_pix_fmt_count_planes(static_cast<AVPixelFormat>(pFrame->format));

  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .layout(torch::kStrided)
                     .device(torch::kCPU);

  torch::Tensor frame = torch::empty({1, num_planes, height, width}, options);
  for (int i = 0; i < num_planes; ++i) {
    torch::Tensor plane = frame.index({0, i});
    uint8_t* tgt = plane.data_ptr<uint8_t>();
    uint8_t* src = pFrame->data[i];
    int linesize = pFrame->linesize[i];
    for (int h = 0; h < height; ++h) {
      memcpy(tgt, src, width);
      tgt += width;
      src += linesize;
    }
  }
  return frame;
}

torch::Tensor convert_yuv420p(AVFrame* pFrame) {
  int width = pFrame->width;
  int height = pFrame->height;

  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .layout(torch::kStrided)
                     .device(torch::kCPU);

  torch::Tensor y = torch::empty({1, height, width, 1}, options);
  {
    uint8_t* tgt = y.data_ptr<uint8_t>();
    uint8_t* src = pFrame->data[0];
    int linesize = pFrame->linesize[0];
    for (int h = 0; h < height; ++h) {
      memcpy(tgt, src, width);
      tgt += width;
      src += linesize;
    }
  }
  torch::Tensor u = torch::empty({1, height / 2, width / 2, 1}, options);
  {
    uint8_t* tgt = u.data_ptr<uint8_t>();
    uint8_t* src = pFrame->data[1];
    int linesize = pFrame->linesize[1];
    for (int h = 0; h < height / 2; ++h) {
      memcpy(tgt, src, width / 2);
      tgt += width / 2;
      src += linesize;
    }
  }
  torch::Tensor v = torch::empty({1, height / 2, width / 2, 1}, options);
  {
    uint8_t* tgt = v.data_ptr<uint8_t>();
    uint8_t* src = pFrame->data[2];
    int linesize = pFrame->linesize[2];
    for (int h = 0; h < height / 2; ++h) {
      memcpy(tgt, src, width / 2);
      tgt += width / 2;
      src += linesize;
    }
  }
  torch::Tensor uv = torch::cat({u, v}, -1);
  // Upsample width and height
  uv = uv.repeat_interleave(2, -2).repeat_interleave(2, -3);
  torch::Tensor t = torch::cat({y, uv}, -1);
  return t.permute({0, 3, 1, 2}); // NCHW
}

torch::Tensor convert_nv12_cpu(AVFrame* pFrame) {
  int width = pFrame->width;
  int height = pFrame->height;

  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .layout(torch::kStrided)
                     .device(torch::kCPU);

  torch::Tensor y = torch::empty({1, height, width, 1}, options);
  {
    uint8_t* tgt = y.data_ptr<uint8_t>();
    uint8_t* src = pFrame->data[0];
    int linesize = pFrame->linesize[0];
    for (int h = 0; h < height; ++h) {
      memcpy(tgt, src, width);
      tgt += width;
      src += linesize;
    }
  }
  torch::Tensor uv = torch::empty({1, height / 2, width / 2, 2}, options);
  {
    uint8_t* tgt = uv.data_ptr<uint8_t>();
    uint8_t* src = pFrame->data[1];
    int linesize = pFrame->linesize[1];
    for (int h = 0; h < height / 2; ++h) {
      memcpy(tgt, src, width);
      tgt += width;
      src += linesize;
    }
  }
  // Upsample width and height
  uv = uv.repeat_interleave(2, -2).repeat_interleave(2, -3);
  torch::Tensor t = torch::cat({y, uv}, -1);
  return t.permute({0, 3, 1, 2}); // NCHW
}

#ifdef USE_CUDA
torch::Tensor convert_nv12_cuda(AVFrame* pFrame, const torch::Device& device) {
  int width = pFrame->width;
  int height = pFrame->height;

  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .layout(torch::kStrided)
                     .device(torch::kCUDA)
                     .device_index(device.index());

  torch::Tensor y = torch::empty({1, height, width, 1}, options);
  {
    uint8_t* tgt = y.data_ptr<uint8_t>();
    CUdeviceptr src = (CUdeviceptr)pFrame->data[0];
    int linesize = pFrame->linesize[0];
    TORCH_CHECK(
        cudaSuccess ==
            cudaMemcpy2D(
                (void*)tgt,
                width,
                (const void*)src,
                linesize,
                width,
                height,
                cudaMemcpyDeviceToDevice),
        "Failed to copy Y plane to Cuda tensor.");
  }
  torch::Tensor uv = torch::empty({1, height / 2, width / 2, 2}, options);
  {
    uint8_t* tgt = uv.data_ptr<uint8_t>();
    CUdeviceptr src = (CUdeviceptr)pFrame->data[1];
    int linesize = pFrame->linesize[1];
    TORCH_CHECK(
        cudaSuccess ==
            cudaMemcpy2D(
                (void*)tgt,
                width,
                (const void*)src,
                linesize,
                width,
                height / 2,
                cudaMemcpyDeviceToDevice),
        "Failed to copy UV plane to Cuda tensor.");
  }
  // Upsample width and height
  uv = uv.repeat_interleave(2, -2).repeat_interleave(2, -3);
  torch::Tensor t = torch::cat({y, uv}, -1);
  return t.permute({0, 3, 1, 2}); // NCHW
}
#endif

torch::Tensor convert_image_tensor(
    AVFrame* pFrame,
    const torch::Device& device) {
  // ref:
  // https://ffmpeg.org/doxygen/4.1/filtering__video_8c_source.html#l00179
  // https://ffmpeg.org/doxygen/4.1/decode__video_8c_source.html#l00038
  AVPixelFormat format = static_cast<AVPixelFormat>(pFrame->format);
  switch (format) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
    case AV_PIX_FMT_ARGB:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_ABGR:
    case AV_PIX_FMT_BGRA:
    case AV_PIX_FMT_GRAY8:
      return convert_interlaced_video(pFrame);
    case AV_PIX_FMT_YUV444P:
      return convert_planar_video(pFrame);
    case AV_PIX_FMT_YUV420P:
      return convert_yuv420p(pFrame);
    case AV_PIX_FMT_NV12:
      return convert_nv12_cpu(pFrame);
#ifdef USE_CUDA
    case AV_PIX_FMT_CUDA: {
      AVHWFramesContext* hwctx =
          (AVHWFramesContext*)pFrame->hw_frames_ctx->data;
      AVPixelFormat sw_format = hwctx->sw_format;
      // cuvid decoder (nvdec frontend of ffmpeg) only supports the following
      // output formats
      // https://github.com/FFmpeg/FFmpeg/blob/072101bd52f7f092ee976f4e6e41c19812ad32fd/libavcodec/cuviddec.c#L1121-L1124
      switch (sw_format) {
        case AV_PIX_FMT_NV12:
          return convert_nv12_cuda(pFrame, device);
        case AV_PIX_FMT_P010:
        case AV_PIX_FMT_P016:
          TORCH_CHECK(
              false,
              "Unsupported video format found in CUDA HW: " +
                  std::string(av_get_pix_fmt_name(sw_format)));
        default:
          TORCH_CHECK(
              false,
              "Unexpected video format found in CUDA HW: " +
                  std::string(av_get_pix_fmt_name(sw_format)));
      }
    }
#endif
    default:
      TORCH_CHECK(
          false,
          "Unexpected video format: " +
              std::string(av_get_pix_fmt_name(format)));
  }
}
} // namespace

void VideoBuffer::push_tensor(torch::Tensor t) {
  // the video frames is expected to contain only one frame
  chunks.push_back(t);
  num_buffered_frames += t.size(0);

  if (frames_per_chunk < 0) {
    return;
  }

  // Trim
  int max_frames = num_chunks * frames_per_chunk;
  if (num_buffered_frames > max_frames) {
    TORCH_WARN_ONCE(
        "The number of buffered frames exceeded the buffer size. "
        "Dropping the old frames. "
        "To avoid this, you can set a higher buffer_chunk_size value.");
    torch::Tensor& t = chunks.front();
    num_buffered_frames -= t.size(0);
    chunks.pop_front();
  }
}

void VideoBuffer::push_frame(AVFrame* frame) {
  push_tensor(convert_image_tensor(frame, device));
}

////////////////////////////////////////////////////////////////////////////////
// Modifiers - Pop
////////////////////////////////////////////////////////////////////////////////

using namespace torch::indexing;

c10::optional<torch::Tensor> Buffer::pop_chunk() {
  if (!num_buffered_frames) {
    return c10::optional<torch::Tensor>{};
  }
  if (frames_per_chunk < 0) {
    return c10::optional<torch::Tensor>{pop_all()};
  }
  return c10::optional<torch::Tensor>{pop_one_chunk()};
}

torch::Tensor AudioBuffer::pop_one_chunk() {
  // Audio deque are aligned with `frames_per_chunk`
  torch::Tensor ret = chunks.front();
  chunks.pop_front();
  num_buffered_frames -= ret.size(0);
  return ret;
}

torch::Tensor VideoBuffer::pop_one_chunk() {
  // Video deque contains one frame par one tensor
  std::vector<torch::Tensor> ret;
  while (num_buffered_frames > 0 && ret.size() < frames_per_chunk) {
    torch::Tensor& t = chunks.front();
    ret.push_back(t);
    chunks.pop_front();
    num_buffered_frames -= 1;
  }
  return torch::cat(ret, 0);
}

torch::Tensor Buffer::pop_all() {
  // Note:
  // This method is common to audio/video.
  // In audio case, each Tensor contains multiple frames
  // In video case, each Tensor contains one frame,
  std::vector<torch::Tensor> ret;
  while (chunks.size()) {
    torch::Tensor& t = chunks.front();
    int n_frames = t.size(0);
    ret.push_back(t);
    chunks.pop_front();
    num_buffered_frames -= n_frames;
  }
  return torch::cat(ret, 0);
}

void Buffer::flush() {
  chunks.clear();
}

} // namespace ffmpeg
} // namespace torchaudio
