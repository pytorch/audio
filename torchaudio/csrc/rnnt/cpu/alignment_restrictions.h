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

} // namespace cpu
} // namespace rnnt
} // namespace torchaudio
