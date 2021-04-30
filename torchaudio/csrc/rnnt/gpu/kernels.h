#pragma once

#include <cassert>

#include <torchaudio/csrc/rnnt/gpu/math.cuh>
#include <torchaudio/csrc/rnnt/gpu/kernel_utils.h>

namespace torchaudio {
namespace rnnt {


template <typename DTYPE, typename CAST_DTYPE>
FORCE_INLINE HOST_AND_DEVICE void ComputeLogProbsSparseElement(
    int bTgt,
    int t,
    int u,
    int maxT,
    int maxU,
    int numTargets,
    int blank,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    const CAST_DTYPE* denominators,
    CAST_DTYPE* logProbs,
    const int* wpEnds,
    const int* validRanges=nullptr,
    const int* cellsPerSample=nullptr,
    int H=1,
    bool fusedLogSmax=true) {

  const int& D = numTargets;

  const int bSrc = bTgt / H;
  const int T = srcLengths[bSrc];
  const int U = tgtLengths[bTgt] + 1;

  if (t >= T || u >= U) { // out of boundary.
    return;
  }
  const int* validRangesB = validRanges + (maxU*2) * bTgt;
  int start = validRangesB[2*u];
  int end = validRangesB[2*u + 1];

  // out of boundary for alignment restriction condition
  if (t < start || t > end) {
    return;
  }
  SparseIndexer idxr(maxU, tgtLengths, validRanges, cellsPerSample);
  int idx = idxr(bTgt, t, u);
  logProbs[(idx << 1) + LOG_PROBS_SKIP_IDX] =
      CAST_DTYPE(logits[idx * D + blank]) - denominators[idx];

  if (!fusedLogSmax) {
    logProbs[(idx << 1) + LOG_PROBS_SKIP_IDX] =
      CAST_DTYPE(logits[idx * D + blank]);
  }

  if (u < U - 1) {
    // emit: log_prob(b, t, u).emit() = logits(b, t, u, tgt[u]) - denom(b, t, u).
    int target = targets[Indexer2D(maxU - 1)(bTgt, u)];
    logProbs[(idx << 1) + LOG_PROBS_EMIT_IDX] =
        CAST_DTYPE(logits[idx * D + target]) - denominators[idx];

    if (!fusedLogSmax) {
      logProbs[(idx << 1) + LOG_PROBS_EMIT_IDX] =
        CAST_DTYPE(logits[idx * D + target]);
    }
  }
}


template <typename DTYPE, typename CAST_DTYPE>
HOST_AND_DEVICE void ComputeGradientsElement(
    int bTgt,
    int t,
    int u,
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
    bool sparse = false,
    const int* validRanges = nullptr,
    const int* cellsPerSample = nullptr,
    int H = 1,
    bool fusedLogSmax = true) {

  const int& maxT = maxSrcLen;
  const int& maxU = maxTgtLen;
  const int& D = numTargets;

  const int bSrc = bTgt / H;
  const int T = srcLengths[bSrc];
  const int U = tgtLengths[bTgt] + 1;

  if (t >= T || u >= U) { // out of boundary.
    if (sparse) {
      // no extra elements needs to be set to 0
      return;
    } else if (gradients == logits && t < maxT && u < maxU) {
      // gradients and logits are pointing to the same memory location
      Indexer3D idxr3(maxT, maxU);
      int idx_b_t_u_zero = idxr3(bTgt, t, u);
      if (idx_b_t_u_zero != -1 ) {
        int start = idx_b_t_u_zero * D;
        for (int b_t_u_d = start; b_t_u_d < start + D; ++b_t_u_d) {
          gradients[b_t_u_d] = 0;
        }
      }
    }
    return;
  }

  int costIdx;
  if (!sparse) {
    costIdx = bTgt * maxT * maxU;
  } else {
    SparseIndexer idxr(maxU, tgtLengths, validRanges, cellsPerSample);
    costIdx = idxr(bTgt, 0, 0);
  }
  CAST_DTYPE cost = -(betas[costIdx]);


  Indexer2D idxr2(maxU - 1);

  int idx_b_t_u, idx_b_t_up1, idx_b_tp1_u, idx_b_tp1_up1;
  if (sparse) {
    SparseIndexer idxr(maxU, tgtLengths, validRanges, cellsPerSample);
    idx_b_t_u = idxr(bTgt, t, u);
    idx_b_t_up1 = idxr(bTgt, t, u+1);
    idx_b_tp1_u = idxr(bTgt, t+1, u);
    idx_b_tp1_up1 = idxr(bTgt, t+1, u+1);
  } else {
    Indexer3D idxr3(maxT, maxU);
    idx_b_t_u = idxr3(bTgt, t, u);
    idx_b_t_up1 = idxr3(bTgt, t, u+1);
    idx_b_tp1_u = idxr3(bTgt, t+1, u);
    idx_b_tp1_up1 = idxr3(bTgt, t+1, u+1);
  }

  if (idx_b_t_u == -1 ) {
    return;
  }

  if (isinf(cost) || isnan(cost)) {
    for (int d = 0; d < D; ++d) {
      int b_t_u_d = idx_b_t_u * D + d;
      gradients[b_t_u_d] = 0;
    }
    return;
  }

  CAST_DTYPE c = alphas[idx_b_t_u] + cost - denominators[idx_b_t_u];
  for (int d = 0; d < D; ++d) {
    int b_t_u_d = idx_b_t_u * D + d;
    CAST_DTYPE g = CAST_DTYPE(logits[b_t_u_d]) + c;

    if (fusedLogSmax) {
      if (d == blank && t == T - 1 && u == U - 1) {  // last blank transition.
        gradients[b_t_u_d] = std::exp(g + betas[idx_b_t_u]) - std::exp(g);
      } else if (t < T - 1 && d == blank) {
        gradients[b_t_u_d] = std::exp(g + betas[idx_b_t_u]);
        if (idx_b_tp1_u != -1) {
          gradients[b_t_u_d] = gradients[b_t_u_d] - std::exp(g + betas[idx_b_tp1_u]);
        }
      } else if (u < U - 1 && d == targets[idxr2(bTgt, u)]) {
        gradients[b_t_u_d] = std::exp(g + betas[idx_b_t_u]);
        if (idx_b_t_up1 != -1) {
          gradients[b_t_u_d] = gradients[b_t_u_d] - std::exp(g + betas[idx_b_t_up1]);
        }
      } else {
        gradients[b_t_u_d] = std::exp(g + betas[idx_b_t_u]);
      }
    } else { // Non fused log softmax case
      CAST_DTYPE g = cost + CAST_DTYPE(logits[b_t_u_d]);
      if (d == blank && t == T - 1 && u == U - 1) {
        gradients[b_t_u_d] = g + alphas[idx_b_t_u];
      } else if (t < T - 1 && d == blank) {
        if (idx_b_tp1_u != -1) {
          gradients[b_t_u_d] = g + alphas[idx_b_t_u] + betas[idx_b_tp1_u];
        } else {
          gradients[b_t_u_d] = g + CAST_DTYPE(-INFINITY);
          }
      } else if (u < U - 1 && d == targets[idxr2(bTgt, u)]) {
        if (idx_b_t_up1 != -1) {
          gradients[b_t_u_d] = g + alphas[idx_b_t_u] + betas[idx_b_t_up1];
        } else {
          gradients[b_t_u_d] = g + CAST_DTYPE(-INFINITY);
        }
      } else {
          gradients[b_t_u_d] = g + CAST_DTYPE(-INFINITY);
      }
      gradients[b_t_u_d] = -std::exp(gradients[b_t_u_d]);
    }

    if (clamp > 0) {
      auto g = CAST_DTYPE(gradients[b_t_u_d]);
      gradients[b_t_u_d] = math::min(g, clamp);
      gradients[b_t_u_d] = math::max(g, -clamp);
    }
  }
}

}  // namespace rnnt
}  // namespace torchaudio
