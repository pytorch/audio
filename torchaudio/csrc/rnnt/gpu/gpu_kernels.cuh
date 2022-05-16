#pragma once

#ifdef USE_CUDA

#include <cassert>
#ifdef __HIP_PLATFORM_AMD__
#include <torchaudio/csrc/rnnt/hip/kernel_utils.h>
#include <torchaudio/csrc/rnnt/hip/kernels.h>
#include <torchaudio/csrc/rnnt/hip/math_hip.cuh>
#else
#include <torchaudio/csrc/rnnt/gpu/kernel_utils.h>
#include <torchaudio/csrc/rnnt/gpu/kernels.h>
#include <torchaudio/csrc/rnnt/gpu/math.cuh>
#endif

namespace torchaudio {
namespace rnnt {

template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeLogProbs(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    const CAST_DTYPE* denominators,
    CAST_DTYPE* logProbs,
    int H = 1) {
  const int& maxT = maxSrcLen;
  const int& maxU = maxTgtLen;
  const int& D = numTargets;

  const int bTgt = blockIdx.z; // 0 <= b < B
  const int bSrc = bTgt / H;
  const int T = srcLengths[bSrc];
  const int U = tgtLengths[bTgt] + 1;

  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int u = blockIdx.y;

  if (t >= T || u >= U) { // out of boundary.
    return;
  }

  Indexer3D indexer(maxT, maxU);

  int idx = indexer(bTgt, t, u);

  // skip: log_prob(b, t, u).skip() = logits(b, t, u, blank) - denom(b, t, u).
  logProbs[(idx << 1) + LOG_PROBS_SKIP_IDX] =
      CAST_DTYPE(logits[idx * D + blank]) - denominators[idx];

  if (u < U - 1) {
    // emit: log_prob(b, t, u).emit() = logits(b, t, u, tgt[u]) - denom(b, t,
    // u).
    int target = targets[Indexer2D(maxU - 1)(bTgt, u)];
    logProbs[(idx << 1) + LOG_PROBS_EMIT_IDX] =
        CAST_DTYPE(logits[idx * D + target]) - denominators[idx];
  }
}

template <typename DTYPE, typename CAST_DTYPE>
__device__ void ComputeAlphas(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* alpha_counters,
    volatile CAST_DTYPE* alphas,
    int H = 1) {
  const int& maxT = maxSrcLen;
  const int& maxU = maxTgtLen;

  const int bTgt = blockIdx.z; // 0 <= b < B
  const int bSrc = bTgt / H;
  const int T = srcLengths[bSrc];
  const int U = tgtLengths[bTgt] + 1;

  const int t = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int u = blockIdx.y + 1;

  if (t >= T || u >= U) { // out of boundary.
    return;
  }

  int* counter = alpha_counters + Indexer2D(maxU)(bTgt, blockIdx.y);

  Indexer3D idxr(maxT, maxU);

  if (t == 1 && u == 1) {
    alphas[idxr(bTgt, 0, 0)] = 0;
  }

  if (blockIdx.x > 0) { // wait for previous warp (in t-axis) is ready.
    while (atomicAdd(counter, 0) < blockIdx.x) {
    }
  }
  if (blockIdx.y > 0) { // wait for previous warp (in u-axis) is ready.
    while (atomicAdd(counter - 1, 0) <= blockIdx.x) {
    }
  }

  if (t == 1 && u < U) {
    // alpha(0, u) = alpha(0, u - 1) + logProbs(0, u - 1).emit().
    alphas[idxr(bTgt, 0, u)] = alphas[idxr(bTgt, 0, u - 1)] +
        logProbs[(idxr(bTgt, 0, u - 1) << 1) + LOG_PROBS_EMIT_IDX];
  }

  if (blockIdx.y == 0 && t < T) {
    CAST_DTYPE skip_prob =
        logProbs[(idxr(bTgt, t - 1, 0) << 1) + LOG_PROBS_SKIP_IDX];
    CAST_DTYPE val;

#pragma unroll
    for (int i = 1; i < warpSize; i <<= 1) {
#ifdef __HIP_PLATFORM_AMD__
      val = __shfl_up(skip_prob, i);
#else
      val = __shfl_up_sync(0xffffffff, skip_prob, i);
#endif
      if (i <= threadIdx.x) {
        skip_prob = skip_prob + val;
      }
    }

    val = alphas[idxr(bTgt, blockIdx.x * blockDim.x, 0)];
    alphas[idxr(bTgt, t, 0)] = skip_prob + val;
  }

  if (t < T && u < U) {
    CAST_DTYPE skip_prob =
        logProbs[(idxr(bTgt, t - 1, u) << 1) + LOG_PROBS_SKIP_IDX];
    CAST_DTYPE emit_prob =
        logProbs[(idxr(bTgt, t, u - 1) << 1) + LOG_PROBS_EMIT_IDX];

    CAST_DTYPE skip =
        alphas[idxr(bTgt, blockIdx.x * blockDim.x, u)] + skip_prob;
    CAST_DTYPE emit = alphas[idxr(bTgt, t, u - 1)] + emit_prob;

    CAST_DTYPE val = math::lse(skip, emit);
    CAST_DTYPE out = val;

    for (int i = 1; i < warpSize; ++i) {
#ifdef __HIP_PLATFORM_AMD__
      val = __shfl_up(val, 1);
#else
      val = __shfl_up_sync(0xffffffff, val, 1);
#endif
      if (i == threadIdx.x) {
        val = math::lse(val + skip_prob, emit);
        out = val;
      }
    }

    alphas[idxr(bTgt, t, u)] = out;
  }

  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(counter, 1);
  }
}

template <typename DTYPE, typename CAST_DTYPE>
__device__ void ComputeBetasCosts(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* betaCounters,
    volatile CAST_DTYPE* betas,
    DTYPE* costs,
    int H = 1) {
  const int& maxT = maxSrcLen;
  const int& maxU = maxTgtLen;

  const int bTgt = blockIdx.z; // 0 <= b < B
  const int bSrc = bTgt / H;
  const int T = srcLengths[bSrc];
  const int U = tgtLengths[bTgt] + 1;

  const int t = T - 2 - blockIdx.x * blockDim.x - threadIdx.x;
  const int u = U - 2 - blockIdx.y;

  if (t < 0 || u < 0) { // out of boundary.
    return;
  }

  int* counter = betaCounters + Indexer2D(maxU)(bTgt, blockIdx.y);

  Indexer3D idxr(maxT, maxU);

  if (t == T - 2 && u == U - 2) {
    betas[idxr(bTgt, T - 1, U - 1)] =
        logProbs[(idxr(bTgt, T - 1, U - 1) << 1) + LOG_PROBS_SKIP_IDX];
  }

  if (blockIdx.x > 0) { // wait for previous warp (in t-axis) is ready.
    while (atomicAdd(counter, 0) < blockIdx.x) {
    }
  }

  if (blockIdx.y > 0) { // wait for previous warp (in u-axis) is ready.
    while (atomicAdd(counter - 1, 0) <= blockIdx.x) {
    }
  }

  if (t == T - 2 && u >= 0) {
    betas[idxr(bTgt, T - 1, u)] = betas[idxr(bTgt, T - 1, u + 1)] +
        logProbs[(idxr(bTgt, T - 1, u) << 1) + LOG_PROBS_EMIT_IDX];
  }

  if (blockIdx.y == 0 && t >= 0) {
    CAST_DTYPE skip_prob =
        logProbs[(idxr(bTgt, t, U - 1) << 1) + LOG_PROBS_SKIP_IDX];
    CAST_DTYPE val;

#pragma unroll
    for (int i = 1; i < warpSize; i <<= 1) {
#ifdef __HIP_PLATFORM_AMD__
      val = __shfl_up(skip_prob, i);
#else
      val = __shfl_up_sync(0xffffffff, skip_prob, i);
#endif
      if (i <= threadIdx.x) {
        skip_prob = skip_prob + val;
      }
    }

    betas[idxr(bTgt, t, U - 1)] =
        betas[idxr(bTgt, T - 1 - blockIdx.x * blockDim.x, U - 1)] + skip_prob;
  }

  if (t >= 0 && u >= 0) {
    CAST_DTYPE skip_prob =
        logProbs[(idxr(bTgt, t, u) << 1) + LOG_PROBS_SKIP_IDX];
    CAST_DTYPE emit_prob =
        logProbs[(idxr(bTgt, t, u) << 1) + LOG_PROBS_EMIT_IDX];

    CAST_DTYPE skip = betas[idxr(bTgt, t + threadIdx.x + 1, u)] + skip_prob;
    CAST_DTYPE emit = betas[idxr(bTgt, t, u + 1)] + emit_prob;

    CAST_DTYPE val = math::lse(skip, emit);
    CAST_DTYPE out = val;

    for (int i = 1; i < warpSize; ++i) {
#ifdef __HIP_PLATFORM_AMD__
      val = __shfl_up(val, 1);
#else
      val = __shfl_up_sync(0xffffffff, val, 1);
#endif
      if (i == threadIdx.x) {
        val = math::lse(val + skip_prob, emit);
        out = val;
      }
    }

    betas[idxr(bTgt, t, u)] = out;

    if (t == 0 && u == 0) { // use -beta(0, 0) as cost.
      costs[bTgt] = DTYPE(-out);
    }
  }

  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(counter, 1);
  }
}

template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeAlphasBetasCosts(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* alpha_counters,
    volatile CAST_DTYPE* alphas,
    int* betaCounters,
    volatile CAST_DTYPE* betas,
    DTYPE* costs,
    int warpSize = 0,
    int numWarps = 0,
    int H = 1) {
  assert(threadIdx.y == 0 || threadIdx.y == 1);

  if (threadIdx.y == 0) {
    ComputeAlphas<DTYPE, CAST_DTYPE>(
        /*maxSrcLen=*/maxSrcLen,
        /*maxTgtLen=*/maxTgtLen,
        /*numTargets=*/numTargets,
        /*blank=*/blank,
        /*logProbs=*/logProbs,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*alpha_counters=*/alpha_counters,
        /*alphas=*/alphas,
        H);
  } else { // threadIdx.y == 1
    ComputeBetasCosts<DTYPE, CAST_DTYPE>(
        /*maxSrcLen=*/maxSrcLen,
        /*maxTgtLen=*/maxTgtLen,
        /*numTargets=*/numTargets,
        /*blank=*/blank,
        /*logProbs=*/logProbs,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*betaCounters=*/betaCounters,
        /*beta=*/betas,
        /*costs=*/costs,
        H);
  }
}

template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeGradients(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    CAST_DTYPE clamp,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    const CAST_DTYPE* denominators,
    const CAST_DTYPE* alphas,
    const CAST_DTYPE* betas,
    DTYPE* gradients,
    int H = 1) {
  const int bTgt = blockIdx.z; // 0 <= b < B
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int u = blockIdx.y;

  ComputeGradientsElement(
      bTgt,
      t,
      u,
      maxSrcLen,
      maxTgtLen,
      numTargets,
      blank,
      clamp,
      logits,
      targets,
      srcLengths,
      tgtLengths,
      denominators,
      alphas,
      betas,
      gradients,
      H);
}

// This is a __global__ wrapper around ComputeAlphas
// device kernel to enable unit testing
template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeAlphasWrapper(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* alpha_counters,
    volatile CAST_DTYPE* alphas,
    int H = 1) {
  ComputeAlphas<DTYPE, CAST_DTYPE>(
      maxSrcLen,
      maxTgtLen,
      numTargets,
      blank,
      logProbs,
      srcLengths,
      tgtLengths,
      alpha_counters,
      alphas,
      H);
}

// This is a __global__ wrapper around ComputeBetas
// device kernel to enable unit testing
template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeBetasWrapper(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* betaCounters,
    volatile CAST_DTYPE* betas,
    DTYPE* costs,
    int H = 1) {
  ComputeBetasCosts<DTYPE, CAST_DTYPE>(
      maxSrcLen,
      maxTgtLen,
      numTargets,
      blank,
      logProbs,
      srcLengths,
      tgtLengths,
      betaCounters,
      betas,
      costs,
      H);
}

// #undef LOG_PROBS_SKIP_IDX
// #undef LOG_PROBS_EMIT_IDX

} // namespace rnnt
} // namespace torchaudio

#endif // USE_CUDA
