#pragma once
#include <vector>
#include <algorithm>

namespace torchaudio {
namespace rnnt {
namespace cpu {

// Helper class which returns locations where
// blank / emit transitions are valid while
// updating alpha / betas for Alignment
// Restricted Transducer loss
class AlignmentRestrictionCheck {
 public:
  AlignmentRestrictionCheck(
      const int* wpEnds,
      int T,
      int U,
      int lBuffer,
      int rBuffer)
      : wpEnds_(wpEnds), T(T), U(U), lBuffer_(lBuffer), rBuffer_(rBuffer) {
  }

  // Returns ranges of valid timesteps which satisfy
  // alignment boundary constraints
  void validTimeRanges(const int u, int& t_start, int& t_end);

  // Examine if doing blank transition into (t, u)
  // is allowed while updating alphas
  // Note that while doing blank transitions for alpha
  // we move from left to right
  bool alphaBlankTransition(const int t, const int u);

  // Examine if doing emit transition into (t, u)
  // is allowed while updating alphas
  // Note that while doing emit transitions for alpha
  // we move from bottom to top
  bool alphaEmitTransition(const int t, const int u);

  // Examine if doing blank transition into (t, u)
  // is allowed while updating betas
  // Note that while doing blank transitions for beta
  // we move from right to left
  bool betaBlankTransition(const int t, const int u);

  // Examine if doing emit transition into (t, u)
  // is allowed while updating betas
  // Note that while doing emit transitions for beta
  // we move from top to bottom
  bool betaEmitTransition(const int t, const int u);

 private:
  const int* wpEnds_;
  int T;
  int U;
  int lBuffer_;
  int rBuffer_;
};

struct SparseIndexer {

  SparseIndexer(
    const int& maxU,
    const int* tgtLengths,
    const int* validRanges,
    const int* cellsPerSample):
    maxU_(maxU),
    tgtLengths_(tgtLengths),
    validRanges_(validRanges),
    cellsPerSample_(cellsPerSample) {};

  int operator() (int bIdx, int tIdx, int uIdx){

      // increment the idx for valid cells in previous samples
      int idx = 0;
      for(int b = 0 ; b < bIdx; b++) {
        idx += cellsPerSample_[b];
      }

      // increment the index for valid cells in current sample
      // using pre-computed valid ranges
      const int* validRangesB = validRanges_ + (maxU_*2) * bIdx;
      int U = tgtLengths_[bIdx] + 1;
      for (int u = 0 ; u < U ; u++) {
        int startT = validRangesB[2 * u];
        int endT = validRangesB[2 * u + 1];
        if (u == uIdx) {
          if (tIdx < startT || tIdx > endT) {
              return -1;
          }
          idx += (tIdx - startT);
          return idx;
        } else {
          idx += (endT - startT) + 1;
        }
      }
      return -1;
  }

  private:
  const int& maxU_;

  // (vector of size B) contains the lengths of U for each sample in batch
  const int* tgtLengths_;

  // (vector of size B*2*U) contains the valid time ranges (start_t, end_t)
  // for each u in every sample, as per alignment restrictions
  const int* validRanges_;

  // The total valid timesteps for each sample (sumed over all u in the sample).
  // This is a quick loopup so we dont need to iterate on all previous u*b values.
  const int* cellsPerSample_;
};

} // namespace cpu
} // namespace rnnt
} // namespace torchaudio
