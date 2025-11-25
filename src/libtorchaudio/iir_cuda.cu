#include <libtorchaudio/utils.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/DeviceGuard.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

template <typename scalar_t>
__global__ void iir_cu_kernel(
    const torchaudio::PackedTensorAccessorSizeT<scalar_t, 3> in,
    const torchaudio::PackedTensorAccessorSizeT<scalar_t, 2> a_flipped,
    torchaudio::PackedTensorAccessorSizeT<scalar_t, 3> out) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t n = in.size(0);
  int64_t c = in.size(1);
  auto total_size = n * c;

  if (tid >= total_size)
    return;

  int64_t n_pos = tid / c;
  int64_t c_pos = tid % c;
  int64_t n_samples_input = in.size(2);
  int64_t n_samples_output = out.size(2);
  int64_t n_order = a_flipped.size(1);

  for (int64_t i = 0; i < n_samples_input; i++) {
    scalar_t a0 = in[n_pos][c_pos][i];
    for (int64_t j = 0; j < n_order - 1; j++)
      a0 -= a_flipped[c_pos][j] * out[n_pos][c_pos][i + j];
    out[n_pos][c_pos][i + n_order - 1] = a0;
  }
}

Tensor cuda_lfilter_core_loop(
    Tensor in,
    Tensor a_flipped,
    Tensor padded_out) {
  STD_TORCH_CHECK(
      in.is_cuda() && a_flipped.is_cuda() &&
      padded_out.is_cuda());

  STD_TORCH_CHECK(
      (in.get_device_index() == a_flipped.get_device_index()) &&
      (in.get_device_index() == padded_out.get_device_index()));
  
  STD_TORCH_CHECK(
      in.is_contiguous() && a_flipped.is_contiguous() &&
      padded_out.is_contiguous());

  STD_TORCH_CHECK(
      (in.scalar_type() == ScalarType::Float || in.scalar_type() == ScalarType::Double) &&
      (a_flipped.scalar_type() == ScalarType::Float ||
       a_flipped.scalar_type() == ScalarType::Double) &&
      (padded_out.scalar_type() == ScalarType::Float ||
       padded_out.scalar_type() == ScalarType::Double));

  const int N = in.size(0);
  const int C = in.size(1);
  STD_TORCH_CHECK(N == padded_out.size(0));
  STD_TORCH_CHECK(C == padded_out.size(1));

  STD_TORCH_CHECK(in.size(2) + a_flipped.size(1) - 1 == padded_out.size(2));

  // TODO: enable device guard:
  // const at::cuda::OptionalCUDAGuard device_guard(in.device());

  const dim3 threads(256);
  const dim3 blocks((N * C + threads.x - 1) / threads.x);

  THO_DISPATCH_V2(
      in.scalar_type(), "iir_cu_loop", AT_WRAP([&] {
        (iir_cu_kernel<scalar_t><<<blocks, threads>>>(
            torchaudio::packed_accessor_size_t<scalar_t, 3>(in),
            torchaudio::packed_accessor_size_t<scalar_t, 2>(a_flipped),
            torchaudio::packed_accessor_size_t<scalar_t, 3>(padded_out)));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        }), AT_FLOATING_TYPES);
  return padded_out;
}
