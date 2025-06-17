#include <type_traits>
#include <string>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

template <typename scalar_t>
__global__ void iir_cu_kernel(
    const torch::
        PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> in,
    const torch::
        PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>
            a_flipped,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>
        out) {
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

void cuda_lfilter_core_loop(
    const torch::Tensor& in,
    const torch::Tensor& a_flipped,
    torch::Tensor& padded_out) {
  TORCH_CHECK(
      in.device().is_cuda() && a_flipped.device().is_cuda() &&
      padded_out.device().is_cuda());

  TORCH_CHECK(
      in.is_contiguous() && a_flipped.is_contiguous() &&
      padded_out.is_contiguous());

  TORCH_CHECK(
      (in.dtype() == torch::kFloat32 || in.dtype() == torch::kFloat64) &&
      (a_flipped.dtype() == torch::kFloat32 ||
       a_flipped.dtype() == torch::kFloat64) &&
      (padded_out.dtype() == torch::kFloat32 ||
       padded_out.dtype() == torch::kFloat64));

  const int N = in.size(0);
  const int C = in.size(1);
  TORCH_CHECK(N == padded_out.size(0));
  TORCH_CHECK(C == padded_out.size(1));

  TORCH_CHECK(in.size(2) + a_flipped.size(1) - 1 == padded_out.size(2));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in));

  const dim3 threads(256);
  const dim3 blocks((N * C + threads.x - 1) / threads.x);

  AT_DISPATCH_FLOATING_TYPES(
      in.scalar_type(), "iir_cu_loop", ([&] {
        iir_cu_kernel<scalar_t><<<blocks, threads>>>(
            in.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            a_flipped.packed_accessor<
                scalar_t,
                2,
                torch::RestrictPtrTraits,
                size_t>(),
            padded_out.packed_accessor<
                scalar_t,
                3,
                torch::RestrictPtrTraits,
                size_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}
