#pragma once

#include <cassert>

#include <torchaudio/csrc/rnnt/cpu/math.h>

namespace torchaudio {
namespace rnnt {

inline HOST_AND_DEVICE bool in_range(
    int start,
    int end, // inclusive
    int val) {
  return start <= val && val <= end;
}

#define LOG_PROBS_SKIP_IDX 0
#define LOG_PROBS_EMIT_IDX 1


struct Indexer2D {
  const int& size2_;

  FORCE_INLINE HOST_AND_DEVICE Indexer2D(const int& size2): size2_(size2) {}

  FORCE_INLINE HOST_AND_DEVICE int operator() (int index1, int index2) {
    return index1 * size2_ + index2;
  }
};


struct Indexer3D {
  const int& size2_;
  const int& size3_;

  FORCE_INLINE HOST_AND_DEVICE Indexer3D(const int& size2, const int& size3)
    : size2_(size2), size3_(size3) {}

  FORCE_INLINE HOST_AND_DEVICE int operator() (int index1, int index2, int index3) {
    return (index1 * size2_ + index2) * size3_ + index3;
  }
};


struct Indexer4D {
  const int& size2_;
  const int& size3_;
  const int& size4_;

  HOST_AND_DEVICE Indexer4D(const int& size2, const int& size3, const int& size4)
    : size2_(size2), size3_(size3), size4_(size4) {}

  HOST_AND_DEVICE int operator() (int index1, int index2, int index3, int index4) {
    return ((index1 * size2_ + index2) * size3_ + index3) * size4_ + index4;
  }
};


struct SparseIndexer {
  HOST_AND_DEVICE SparseIndexer(
    const int& maxU,
    const int* tgtLengths,
    const int* validRanges,
    const int* cellsPerSample) :
    maxU_(maxU),
    tgtLengths_(tgtLengths),
    validRanges_(validRanges),
    cellsPerSample_(cellsPerSample) {};

  HOST_AND_DEVICE int operator() (int bIdx, int tIdx, int uIdx) {
    // Returns the sparse index given bIdx, tIdx, uIdx; or -1 if t out of band.

    // increment the idx for valid cells in previous samples
    // TODO: Inefficient in the inner loop; precompute this?
    int idx = 0;
    for(int b = 0 ; b < bIdx; b++) {
      idx += cellsPerSample_[b];
    }

    // increment the idx for valid cells in current sample
    // using pre-computed valid ranges
    const int* validRangesB = validRanges_ + (maxU_ * 2) * bIdx;
    int U = tgtLengths_[bIdx] + 1;
    for (int u = 0 ; u < U ; u++) {
      int startT = validRangesB[2*u];
      int endT = validRangesB[2*u + 1];
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

}  // namespace rnnt
}  // namespace torchaudio
