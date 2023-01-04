#pragma once
#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace ffmpeg {
namespace detail {

//////////////////////////////////////////////////////////////////////////////
// Helper functions
//////////////////////////////////////////////////////////////////////////////
torch::Tensor convert_audio(AVFrame* frame);

torch::Tensor get_buffer(
    at::IntArrayRef shape,
    const torch::Device& device = torch::Device(torch::kCPU));
torch::Tensor get_image_buffer(AVFrame* pFrame, const torch::Device& device);

void write_yuv420p(AVFrame* pFrame, torch::Tensor& yuv);
void write_nv12_cpu(AVFrame* pFrame, torch::Tensor& yuv);
#ifdef USE_CUDA
void write_nv12_cuda(AVFrame* pFrame, torch::Tensor& yuv);
#endif
void write_interlaced_image(AVFrame* pFrame, torch::Tensor& frame);
void write_planar_image(AVFrame* pFrame, torch::Tensor& frame);
torch::Tensor convert_image(AVFrame* frame, const torch::Device& device);

} // namespace detail
} // namespace ffmpeg
} // namespace torchaudio
