#pragma once

#ifdef USE_CUDA

#include <cassert>

#include <torchaudio/csrc/rnnt/gpu/math.cuh>
#include <torchaudio/csrc/rnnt/gpu/kernels.h>
#include <torchaudio/csrc/rnnt/gpu/kernel_utils.h>

namespace torchaudio {
namespace rnnt {


template <typename DTYPE, typename CAST_DTYPE>
__device__ void ComputeAlphasRestricted(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* alpha_counters,
    volatile CAST_DTYPE* alphas,
    const int* wpEnds,
    int lBuffer,
    int rBuffer,
    int warp_size,
    int H=1) {

    const int& maxT = maxSrcLen;
    const int& maxU = maxTgtLen;

    // Block's 3d dim is batchsize
    const int bTgt = blockIdx.z;  // 0 <= b < B
    const int bSrc = bTgt / H;
    const int T = srcLengths[bSrc];
    const int U = tgtLengths[bTgt] + 1;

    assert(T <= maxT && "T must be < maxT");
    assert(U <= maxU && "U must be < maxU");

    // Blocks 1st dim is T i.e. warpsize timesteps
    // to find timestep, we index into correct block idx * dim
    // then offset by the thread
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    // Blocks 2nd dim is U. For u, we just offset by 2nd idx
    const int u = blockIdx.y;

    // simply exit if we are out of bounds
    // this happens typically when warp size is larger than T
    if (t > T-1 || u > U-1 || t < 0 || u < 0) {
        return;
    }

    const int* wpEndsB = wpEnds + maxU * bTgt;
    assert(wpEndsB[u] >= 0 && "wpEnds cannot be negative");
    if (u < U - 1) {
        assert(wpEndsB[u+1] >= 0 && "wpEnds cannot be negative");
    }

    // below indexes are inclusive
    // compute valid ranges for emit transitions in current warp
    int emit_start = math::max(wpEndsB[u] - lBuffer, 0);
    int emit_end = math::min(wpEndsB[u] + rBuffer, T - 1);

    // compute valid ranges for blank transitions in current warp
    int blank_start = math::max(wpEndsB[u] - lBuffer + 1, 1);
    int blank_end = (u == U - 1) ? T - 1 : math::min(wpEndsB[u + 1] + rBuffer, T - 1);

    // this is a useful pointer to track the last timestep
    // for previous warp.
    int prevWarpEndT = blockIdx.x * blockDim.x - 1;

    int* counter = alpha_counters + Indexer2D(maxU)(bTgt, blockIdx.y);


    int idx_b_t_u = -1, idx_b_t_um1 = -1, idx_b_tm1_u = -1;
    int idx_b_prevWarpEndT_u = -1;

    Indexer3D idxr(maxT, maxU);
    idx_b_t_u = idxr(bTgt, t, u);
    idx_b_t_um1 = idxr(bTgt, t, u - 1);
    idx_b_tm1_u = idxr(bTgt, t - 1, u);
    idx_b_prevWarpEndT_u = idxr(bTgt, prevWarpEndT, u);

    alphas[idx_b_t_u] = CAST_DTYPE(-INFINITY);

    // Initialization condition for alphas
    if (t == 0 && u == 0) {
        alphas[idx_b_t_u] = 0;
    }

    // Synchronization routines:
    // Below conditions make sure given t, u
    // - threads in previous warp for u is complete
    // - threads in the warp corresponding to t, u-1 is complete
    // See https://github.com/1ytic/warp-rnnt for illustration
    if (blockIdx.x > 0) {  // wait for previous warp (in t-axis) is ready.
        while (atomicAdd(counter, 0) < blockIdx.x) {}
    }

    if (blockIdx.y > 0) {  // wait for previous warp (in u-axis) is ready.
        while (atomicAdd(counter - 1, 0) <= blockIdx.x) {}
    }

    // This is initialization loop for t = 0, u
    // i.e. emit condition while running alphas
    if (t == 0 && u > 0) {
        // Alignment Restriction check
        if (in_range(emit_start, emit_end, t)) {
            alphas[idx_b_t_u] =
            alphas[idx_b_t_um1]
            + logProbs[(idx_b_t_um1 << 1) + LOG_PROBS_EMIT_IDX];
        }
    }

    // This is initialization loop for u = 0, t
    if (u == 0 && t > 0) {

        // skip_prob stores the log probability of t, u
        CAST_DTYPE skip_prob = logProbs[(idx_b_tm1_u << 1) + LOG_PROBS_SKIP_IDX];

        #pragma unroll
        // iterating on i = 1, 2, 4, 8..
        for (int i = 1; i < warp_size; i <<= 1) {

            // https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
            // first param is a mask -1 indicates all threads in warp participate
            // second param is the value being passed for current thread
            // third param is the offset which will be used for addition of values
            // copy from a thread with lower ID relative to the caller
            CAST_DTYPE synced_val = __shfl_up_sync(0xffffffff, skip_prob, i);

            // Illustration with 8 threads in warp
            // Initial:   0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
            // i == 1:    0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
            // i == 2:    0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4
            // i == 4:    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8

            // Alignment restriction check
            // add to skip_prob only if *t-i* is within allowed range
            if (i <= threadIdx.x &&
                in_range(blank_start, blank_end, t - i) ) {
                skip_prob += synced_val;
            }
        }

        // update the values of alpha if *t* is within allowed range
        if (in_range(blank_start, blank_end, t)) {
            alphas[idx_b_t_u] = skip_prob;
        }

        // Optionally add previous warp's end idx value if it is
        // within range to get cumulative score for all previous timesteps,
        // and not just within current warp
        if (in_range(blank_start, blank_end, prevWarpEndT)) {
            alphas[idx_b_t_u] += alphas[idx_b_prevWarpEndT_u];
        }
    }

    // General case
    if (t < T && u < U) {

        CAST_DTYPE skip_prob = CAST_DTYPE(-INFINITY);
        if (in_range(blank_start, blank_end, t) && t > 0) {
            skip_prob = logProbs[(idx_b_tm1_u << 1) + LOG_PROBS_SKIP_IDX];
        }

        // We check if the first index of current warp (prevWarpEndT+1) was within allowed
        // timesteps, if so we add the alphas of previous warp's last timestep
        // to skip score.
        CAST_DTYPE skip = skip_prob;
        if (in_range(blank_start, blank_end, prevWarpEndT+1) && prevWarpEndT >= 0) {
            skip += alphas[idx_b_prevWarpEndT_u];
        }

        CAST_DTYPE emit = CAST_DTYPE(-INFINITY);
        if (in_range(emit_start, emit_end, t) && u > 0) {
            emit = alphas[idx_b_t_um1] +
                logProbs[(idx_b_t_um1 << 1) + LOG_PROBS_EMIT_IDX];
        }

        CAST_DTYPE out_score = math::lse(skip, emit);


        // Below we loop over warp_size updating the out score
        // once per thread in each loop.
        // shared memory is used to synchronize the values in threads
        for(int i = 1; i < warp_size; i++) {

            CAST_DTYPE synced_val = __shfl_up_sync(0xffffffff, out_score, 1);

            // We will only update the out_score if t is within range
            // of alignment constraints and it is the thread's turn in loop
            if ((i == threadIdx.x) && in_range(blank_start, blank_end, t)) {
                out_score = math::lse(synced_val + skip_prob, emit);
            }
        }
        // update alphas if t was within range of blanks, or emits
        if ( (in_range(blank_start, blank_end, t) || in_range(emit_start, emit_end, t) )
            && (t > 0 && u > 0)) {
            alphas[idx_b_t_u] = out_score;
        }
    }

    // Synchronization mechanism
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(counter, 1);
    }
}



// This is a wrapper around ComputeAlphasRestricted
// kernel to enable unit testing
template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeAlphasRestrictedWrapper(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* alpha_counters,
    volatile CAST_DTYPE* alphas,
    const int* wpEnds,
    int lBuffer,
    int rBuffer,
    int warp_size,
    int H=1) {
    ComputeAlphasRestricted<DTYPE, CAST_DTYPE>(
      maxSrcLen,
      maxTgtLen,
      numTargets,
      blank,
      logProbs,
      srcLengths,
      tgtLengths,
      alpha_counters,
      alphas,
      wpEnds,
      lBuffer,
      rBuffer,
      warp_size,
      H);
}


template <typename DTYPE, typename CAST_DTYPE>
__device__ void ComputeBetasCostsRestricted(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* beta_counters,
    DTYPE* costs,
    volatile CAST_DTYPE* betas,
    const int* wpEnds,
    int lBuffer,
    int rBuffer,
    int warp_size,
    int num_warps,
    int H=1) {

    const int& maxT = maxSrcLen;
    const int& maxU = maxTgtLen;

    // Block's 3d dim is batchsize
    const int bTgt = blockIdx.z;  // 0 <= b < B
    const int bSrc = bTgt / H;
    const int T = srcLengths[bSrc];
    const int U = tgtLengths[bTgt] + 1;

    assert(T <= maxT);
    assert(U <= maxU);

    const int t = T - 1 - blockIdx.x * blockDim.x - threadIdx.x;
    const int u = U - 1 - blockIdx.y;

    if (t < 0 || u < 0) { // out of boundary.
        return;
    }

    const int* wpEndsB = wpEnds + maxU * bTgt;

    // below indexes are inclusive
    // compute valid ranges for emit transitions in
    // current warp
    int emit_start = (u >= U - 1) ? -1 : math::max(0, wpEndsB[u + 1] - lBuffer);
    int emit_end = (u >= U - 1) ? -1 : math::min(wpEndsB[u + 1] + rBuffer, T - 1);


    // compute valid ranges for blank transitions in
    // current warp
    int blank_start = (u > U - 1) ? -1 : math::max(wpEndsB[u] - lBuffer, 0);
    int blank_end = (u >= U - 1) ? T - 2 : math::min(wpEndsB[u + 1] + rBuffer - 1, T - 2);

    int rightBlockStartT = T - 1 - blockIdx.x * blockDim.x + 1;
    int* counter = beta_counters + Indexer2D(maxU)(bTgt, blockIdx.y);

    int idx_b_t_u = -1, idx_b_t_up1 = -1;
    int idx_b_tp1_up1 = -1, idx_b_rightBlockStartT_u = -1;

    Indexer3D idxr(maxT, maxU);
    idx_b_t_u = idxr(bTgt, t, u);
    idx_b_t_up1 = idxr(bTgt, t, u+1);
    idx_b_tp1_up1 = idxr(bTgt, t+1, u+1);
    idx_b_rightBlockStartT_u = idxr(bTgt, rightBlockStartT, u);

    if (t < T && u < U) {
        betas[idx_b_t_u] = CAST_DTYPE(-INFINITY);
    }

    // Initialization condition
    if (t == T - 1 && u == U - 1) {
        betas[idx_b_t_u] =
            logProbs[(idx_b_t_u << 1) + LOG_PROBS_SKIP_IDX];
    }

    // For betas, we process warps from right to left
    // blockIdx.x specifies the warp id for a given u
    // here we want to wait until the warp to the right is done
    if (blockIdx.x > 0) {  // wait for previous warp (in t-axis) is ready.
        while (atomicAdd(counter, 0) < blockIdx.x) {}
    }

    if (blockIdx.y > 0) {  // wait for previous warp (in u-axis) is ready.
        while (atomicAdd(counter - 1, 0) <= blockIdx.x) {}
    }

    // edge condition for emits
    if (t == T - 1 && u < U - 1) {
        if (emit_start <= t && t <= emit_end) {
            betas[idx_b_t_u] =
            betas[idx_b_t_up1]
            + logProbs[(idx_b_t_u << 1) + LOG_PROBS_EMIT_IDX];
        }
    }

    if (u == U - 1 && t <= T - 1) {
        CAST_DTYPE skip_prob = logProbs[(idx_b_t_u << 1) + LOG_PROBS_SKIP_IDX];

        #pragma unroll
        for(int i = 1; i < warp_size; i <<= 1) {

            CAST_DTYPE synced_val = __shfl_up_sync(0xffffffff, skip_prob, i);
            if (i <= threadIdx.x && in_range(blank_start, blank_end, t)) {
                skip_prob = skip_prob + synced_val;
            }
        }

        if (in_range(blank_start, blank_end, t) && t < T - 1) {
            // we add skipprob and right block's first value
            // and assign it to beta
            betas[idx_b_t_u]  = skip_prob;
            if (in_range(blank_start, blank_end, rightBlockStartT) ||
                in_range(emit_start, emit_end, rightBlockStartT)) {
                betas[idx_b_t_u] += betas[idx_b_rightBlockStartT_u];
            }
        }
    }

    // General case
    if (t < T && u < U) {

        CAST_DTYPE skip_prob = CAST_DTYPE(-INFINITY);
        if (in_range(blank_start, blank_end, t)) {
            skip_prob = logProbs[(idx_b_t_u << 1) + LOG_PROBS_SKIP_IDX];
        }
        // We check if the first index of right warp is within allowed
        // timesteps, if so we add the betas of right warp's first timestep
        // to skip score
        CAST_DTYPE skip = skip_prob;
        if (in_range(blank_start, blank_end, rightBlockStartT) || in_range(emit_start, emit_end, rightBlockStartT)) {
            skip += betas[idx_b_rightBlockStartT_u];
        }

        CAST_DTYPE emit = CAST_DTYPE(-INFINITY);
        if (in_range(emit_start, emit_end, t) && u < U - 1) {
            emit = betas[idx_b_t_up1] +
                logProbs[(idx_b_t_u << 1) + LOG_PROBS_EMIT_IDX];
        }

        CAST_DTYPE out_score = CAST_DTYPE(-INFINITY);
        if (skip != CAST_DTYPE(-INFINITY) || emit != CAST_DTYPE(-INFINITY)) {
            out_score = math::lse(skip, emit);
        }

        for(int i = 1; i < warp_size; i++) {

            CAST_DTYPE synced_val = __shfl_up_sync(0xffffffff, out_score, 1);

            if (i == threadIdx.x &&
                (in_range(blank_start, blank_end, t) ||
                in_range(emit_start, emit_end, t))
            ) {
                out_score = math::lse(synced_val + skip_prob, emit);
            }
        }

        // bounds check for last warps
        if (t < T - 1 && u < U - 1) {
            // bound checks for t
            if (in_range(blank_start, blank_end, t) ||
                    in_range(emit_start, emit_end, t)) {
                betas[idx_b_t_u] = out_score;
            }
        }
    }

    if (t == 0 && u == 0) {  // use -beta(0, 0) as cost.
        if (isinf(-betas[idx_b_t_u]) || isnan(-betas[idx_b_t_u])) {
            costs[bTgt] = 0;
        } else {
            costs[bTgt] = -betas[idx_b_t_u];
        }
    }

    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(counter, 1);
    }
}


// This is a wrapper around ComputeAlphasRestricted
// kernel to enable unit testing
template <typename DTYPE, typename CAST_DTYPE>
__global__ void ComputeBetasCostsRestrictedWrapper(
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    const CAST_DTYPE* logProbs,
    const int* srcLengths,
    const int* tgtLengths,
    int* beta_counters,
    CAST_DTYPE* costs,
    volatile CAST_DTYPE* betas,
    const int* wpEnds,
    int lBuffer,
    int rBuffer,
    int warp_size,
    int num_warp,
    int H=1) {
    ComputeBetasCostsRestricted<DTYPE, CAST_DTYPE>(
      maxSrcLen,
      maxTgtLen,
      numTargets,
      blank,
      logProbs,
      srcLengths,
      tgtLengths,
      beta_counters,
      costs,
      betas,
      wpEnds,
      lBuffer,
      rBuffer,
      warp_size,
      num_warp,
      H);
}

}  // namespace rnnt
}  // namespace torchaudio

#endif  // USE_CUDA
