#include <torchaudio/csrc/rnnt/cpu/alignment_restrictions.h>

namespace torchaudio {
namespace rnnt {
namespace cpu {

void AlignmentRestrictionCheck::validTimeRanges(
    const int u,
    int& t_start,
    int& t_end) {
  t_start = std::max(wpEnds_[u] - lBuffer_, 0);
  t_end = (u == U - 1) ? T - 1 : std::min(wpEnds_[u + 1] + rBuffer_, T - 1);
  return;
}

bool AlignmentRestrictionCheck::alphaBlankTransition(
    const int t,
    const int u) {
  if (u == 0 && t == 0) {
    return false;
  }

  // for alpha blank updates we move from left to right
  // blank transitions are valid from:
  // start time when current symbol is emitted
  // **offset to right by 1**
  int start = std::max(wpEnds_[u] - lBuffer_ + 1, 1);

  // blank transitions are valid until:
  // for U-1: last allowed timestep i.e. T - 1
  // for other cases: last time we may emit the next symbol
  int end = (u == U - 1) ? T - 1 : std::min(wpEnds_[u + 1] + rBuffer_, T - 1);

  return start <= t && t <= end;
}

bool AlignmentRestrictionCheck::alphaEmitTransition(
    const int t,
    const int u) {
  // For alphas we move from bottom to top
  // emit transitions into (t, 0) for alpha are invalid
  if (u == 0) {
    return false;
  }

  // For alphas emit are valid starting from:
  // first time when current symbol can be emitted
  int start = std::max(wpEnds_[u] - lBuffer_, 0);

  // For alphas emit are valid until:
  // last timestep when current symbol can be emitted
  int end = std::min(wpEnds_[u] + rBuffer_, T - 1);

  return start <= t && t <= end;
}

bool AlignmentRestrictionCheck::betaBlankTransition(
    const int t,
    const int u) {
  // For updating betas with blank transition
  // we move from right to left

  // for beta, blanks transitions are can start
  // first timestep when we emit previous symbol
  int start = std::max(wpEnds_[u] - lBuffer_, 0);

  // for beta, blanks transitions are valid until
  // we can emit current symbol **offset to left by 1**
  // note: T-2, we init beta[-1, -1] by log_prob[-1, -1, blank]
  int end =
      (u == U - 1) ? T - 2 : std::min(wpEnds_[u + 1] + rBuffer_ - 1, T - 2);

  return start <= t && t <= end;
}

bool AlignmentRestrictionCheck::betaEmitTransition(const int t, const int u) {
  // While updating betas we go from top to bottom.

  // for last symbol, we do not allow emit transition,
  // so beta into u-1, t is invalid
  if (u == U - 1) {
    return false;
  }

  // For betas we allow emit transition starting from
  // first time we can emit next symbol
  int start = std::max(0, wpEnds_[u + 1] - lBuffer_);

  // For betas we allow emit transitions to end with
  // last time we can emit next symbol
  int end = std::min(wpEnds_[u + 1] + rBuffer_, T - 1);

  return start <= t and t <= end;
}

} // namespace cpu
} // namespace rnnt
} // namespace torchaudio
