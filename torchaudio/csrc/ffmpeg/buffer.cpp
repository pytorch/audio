#include <torchaudio/csrc/ffmpeg/buffer.h>
#include <stdexcept>
#include <vector>

namespace torchaudio {
namespace ffmpeg {

Buffer::Buffer(int frames_per_chunk, int num_chunks)
    : frames_per_chunk(frames_per_chunk), num_chunks(num_chunks) {}

AudioBuffer::AudioBuffer(int frames_per_chunk, int num_chunks)
    : Buffer(frames_per_chunk, num_chunks) {}

VideoBuffer::VideoBuffer(int frames_per_chunk, int num_chunks)
    : Buffer(frames_per_chunk, num_chunks) {}

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
      throw std::runtime_error(
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
torch::Tensor convert_image_tensor(AVFrame* pFrame) {
  // ref:
  // https://ffmpeg.org/doxygen/4.1/filtering__video_8c_source.html#l00179
  // https://ffmpeg.org/doxygen/4.1/decode__video_8c_source.html#l00038
  AVPixelFormat format = static_cast<AVPixelFormat>(pFrame->format);
  int width = pFrame->width;
  int height = pFrame->height;
  uint8_t* buf = pFrame->data[0];
  int linesize = pFrame->linesize[0];

  int channel;
  switch (format) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
      channel = 3;
      break;
    case AV_PIX_FMT_ARGB:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_ABGR:
    case AV_PIX_FMT_BGRA:
      channel = 4;
      break;
    case AV_PIX_FMT_GRAY8:
      channel = 1;
      break;
    default:
      throw std::runtime_error(
          "Unexpected format: " + std::string(av_get_pix_fmt_name(format)));
  }

  torch::Tensor t;
  t = torch::empty({1, height, width, channel}, torch::kUInt8);
  auto ptr = t.data_ptr<uint8_t>();
  int stride = width * channel;
  for (int i = 0; i < height; ++i) {
    memcpy(ptr, buf, stride);
    buf += linesize;
    ptr += stride;
  }
  return t.permute({0, 3, 1, 2});
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
    torch::Tensor& t = chunks.front();
    num_buffered_frames -= t.size(0);
    chunks.pop_front();
  }
}

void VideoBuffer::push_frame(AVFrame* frame) {
  push_tensor(convert_image_tensor(frame));
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

} // namespace ffmpeg
} // namespace torchaudio
