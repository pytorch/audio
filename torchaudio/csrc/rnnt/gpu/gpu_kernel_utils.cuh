#pragma once

#ifdef USE_CUDA

#ifdef __HIP_PLATFORM_AMD__
#include <torchaudio/csrc/rnnt/hip/math_hip.cuh>
#else
#include <torchaudio/csrc/rnnt/gpu/math.cuh>
#endif

namespace torchaudio {
namespace rnnt {

template <int NUM_THREADS, typename DTYPE, typename CAST_DTYPE>
__global__ void ReduceMax2D(
    int dim,
    const DTYPE* inputs, // [N, dim]
    CAST_DTYPE* outputs) {
  __shared__ CAST_DTYPE shared[NUM_THREADS];

  // each thread reduces one matrix row
  int offset = blockIdx.x * dim; // [n, 0]
  CAST_DTYPE val = inputs[offset]; // default = inputs(n, 0)
  for (int d = threadIdx.x; d < dim; d += NUM_THREADS) {
    CAST_DTYPE next = inputs[offset + d];
    if (next > val) {
      val = next;
    }
  }

  shared[threadIdx.x] = val;
  __syncthreads();

  for (int stride = (NUM_THREADS >> 1); stride >= WARP_SIZE; stride >>= 1) {
    if (threadIdx.x < stride && threadIdx.x + stride < dim) {
      if (shared[threadIdx.x + stride] > shared[threadIdx.x]) {
        shared[threadIdx.x] = shared[threadIdx.x + stride];
        val = shared[threadIdx.x];
      }
    }
    __syncthreads();
  }

  CAST_DTYPE shf;
  for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1) {
#ifdef __HIP_PLATFORM_AMD__
    shf = __shfl_down(val, stride);
#else
    shf = __shfl_down_sync(0xFFFFFFFF, val, stride);
#endif
    if (threadIdx.x < stride && threadIdx.x + stride < dim) {
      if (shf > val) {
        val = shf;
      }
    }
  }

  if (threadIdx.x == 0) {
    outputs[blockIdx.x] = val;
  }
}

template <int NUM_THREADS, typename DTYPE, typename CAST_DTYPE>
__global__ void ReduceLogSumExpGivenMax2D(
    int dim,
    const DTYPE* inputs, // [N, dim]
    CAST_DTYPE* outputs) { // in: max -> out: logsum

  __shared__ CAST_DTYPE shared[NUM_THREADS];

  CAST_DTYPE max = outputs[blockIdx.x];
  CAST_DTYPE val = 0;

  int offset = blockIdx.x * dim;
  for (int d = threadIdx.x; d < dim; d += NUM_THREADS) {
    val = val + std::exp(CAST_DTYPE(inputs[offset + d]) - max);
  }

  shared[threadIdx.x] = val;
  __syncthreads();

  for (int stride = (NUM_THREADS >> 1); stride >= WARP_SIZE; stride >>= 1) {
    if (threadIdx.x < stride && threadIdx.x + stride < dim) {
      val = shared[threadIdx.x] + shared[threadIdx.x + stride];
      shared[threadIdx.x] = val;
    }
    __syncthreads();
  }

  CAST_DTYPE shf;
  for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1) {
#ifdef __HIP_PLATFORM_AMD__
    shf = __shfl_down(val, stride);
#else
    shf = __shfl_down_sync(0xFFFFFFFF, val, stride);
#endif
    if (threadIdx.x < stride && threadIdx.x + stride < dim) {
      val = val + shf;
    }
  }

  if (threadIdx.x == 0) {
    outputs[blockIdx.x] = max + std::log(val);
  }
}

} // namespace rnnt
} // namespace torchaudio

#endif // USE_CUDA
