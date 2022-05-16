#pragma once

#include <cassert>

#ifdef __HIP_PLATFORM_AMD__
#include <torchaudio/csrc/rnnt/hip/kernel_utils.h>
#include <torchaudio/csrc/rnnt/hip/math_hip.cuh>
#else
#include <torchaudio/csrc/rnnt/gpu/kernel_utils.h>
#include <torchaudio/csrc/rnnt/gpu/math.cuh>
#endif

namespace torchaudio {
namespace rnnt {

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
    int H = 1) {
  const int& maxT = maxSrcLen;
  const int& maxU = maxTgtLen;
  const int& D = numTargets;

  const int bSrc = bTgt / H;
  const int T = srcLengths[bSrc];
  const int U = tgtLengths[bTgt] + 1;

  if (t >= T || u >= U) { // out of boundary.
    if (gradients == logits && t < maxT && u < maxU) {
      // gradients and logits are pointing to the same memory location
      Indexer3D idxr3(maxT, maxU);
      int idx_b_t_u_zero = idxr3(bTgt, t, u);
      if (idx_b_t_u_zero != -1) {
        int start = idx_b_t_u_zero * D;
        for (int b_t_u_d = start; b_t_u_d < start + D; ++b_t_u_d) {
          gradients[b_t_u_d] = 0;
        }
      }
    }
    return;
  }

  int costIdx = bTgt * maxT * maxU;
  CAST_DTYPE cost = -(betas[costIdx]);

  Indexer2D idxr2(maxU - 1);

  int idx_b_t_u, idx_b_t_up1, idx_b_tp1_u;
  Indexer3D idxr3(maxT, maxU);
  idx_b_t_u = idxr3(bTgt, t, u);
  idx_b_t_up1 = idxr3(bTgt, t, u + 1);
  idx_b_tp1_u = idxr3(bTgt, t + 1, u);

  if (idx_b_t_u == -1) {
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

    if (d == blank && t == T - 1 && u == U - 1) { // last blank transition.
      gradients[b_t_u_d] = std::exp(g + betas[idx_b_t_u]) - std::exp(g);
    } else if (t < T - 1 && d == blank) {
      gradients[b_t_u_d] = std::exp(g + betas[idx_b_t_u]);
      if (idx_b_tp1_u != -1) {
        gradients[b_t_u_d] =
            gradients[b_t_u_d] - std::exp(g + betas[idx_b_tp1_u]);
      }
    } else if (u < U - 1 && d == targets[idxr2(bTgt, u)]) {
      gradients[b_t_u_d] = std::exp(g + betas[idx_b_t_u]);
      if (idx_b_t_up1 != -1) {
        gradients[b_t_u_d] =
            gradients[b_t_u_d] - std::exp(g + betas[idx_b_t_up1]);
      }
    } else {
      gradients[b_t_u_d] = std::exp(g + betas[idx_b_t_u]);
    }

    if (clamp > 0) {
      auto g = CAST_DTYPE(gradients[b_t_u_d]);
      gradients[b_t_u_d] = math::min(g, clamp);
      gradients[b_t_u_d] = math::max(g, -clamp);
    }
  }
}

} // namespace rnnt
} // namespace torchaudio
