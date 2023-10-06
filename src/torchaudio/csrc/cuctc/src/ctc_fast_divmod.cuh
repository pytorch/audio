/**
 *  Modified from NVIDIA/cutlass(https://github.com/NVIDIA/cutlass)
 *
 */

#pragma once

namespace cu_ctc {
template <typename value_t>
__host__ __device__ __forceinline__ value_t clz(value_t x) {
  for (int i = 31; i >= 0; --i) {
    if ((1 << i) & x)
      return 31 - i;
  }
  return 32;
}

template <typename value_t>
__host__ __device__ __forceinline__ value_t find_log2(value_t x) {
  int a = int(31 - clz(x));
  a += (x & (x - 1)) != 0; // Round up, add 1 if not a power of 2.
  return a;
}

/**
 * Find divisor, using find_log2
 */
__host__ __device__ __forceinline__ void find_divisor(
    unsigned int& mul,
    unsigned int& shr,
    unsigned int denom) {
  if (denom == 1) {
    mul = 0;
    shr = 0;
  } else {
    unsigned int p = 31 + find_log2(denom);
    unsigned m =
        unsigned(((1ull << p) + unsigned(denom) - 1) / unsigned(denom));

    mul = m;
    shr = p - 32;
  }
}

__host__ __device__ __forceinline__

    void
    fast_divmod(
        int& quo,
        int& rem,
        int src,
        int div,
        unsigned int mul,
        unsigned int shr) {
#if defined(__CUDA_ARCH__)
  // Use IMUL.HI if div != 1, else simply copy the source.
  quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
#else
  quo = int((div != 1) ? int(((int64_t)src * mul) >> 32) >> shr : src);
#endif

  // The remainder.
  rem = src - (quo * div);
}

// For long int input
__host__ __device__ __forceinline__ void fast_divmod(
    int& quo,
    int64_t& rem,
    int64_t src,
    int div,
    unsigned int mul,
    unsigned int shr) {
#if defined(__CUDA_ARCH__)
  // Use IMUL.HI if div != 1, else simply copy the source.
  quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
#else
  quo = int((div != 1) ? ((src * mul) >> 32) >> shr : src);
#endif
  // The remainder.
  rem = src - (quo * div);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Object to encapsulate the fast division+modulus operation.
///
/// This object precomputes two values used to accelerate the computation and is
/// best used when the divisor is a grid-invariant. In this case, it may be
/// computed in host code and marshalled along other kernel arguments using the
/// 'Params' pattern.
///
/// Example:
///
///
///   int quotient, remainder, dividend, divisor;
///
///   FastDivmod divmod(divisor);
///
///   divmod(quotient, remainder, dividend);
///
///   // quotient = (dividend / divisor)
///   // remainder = (dividend % divisor)
///
struct FastDivmod {
  int divisor;
  unsigned int multiplier;
  unsigned int shift_right;

  /// Construct the FastDivmod object, in host code ideally.
  ///
  /// This precomputes some values based on the divisor and is computationally
  /// expensive.

  __host__ __device__ __forceinline__ FastDivmod()
      : divisor(0), multiplier(0), shift_right(0) {}

  __host__ __device__ __forceinline__ FastDivmod(int divisor_)
      : divisor(divisor_) {
    find_divisor(multiplier, shift_right, divisor);
  }

  /// Computes integer division and modulus using precomputed values. This is
  /// computationally inexpensive.
  __host__ __device__ __forceinline__ void operator()(
      int& quotient,
      int& remainder,
      int dividend) const {
    fast_divmod(
        quotient, remainder, dividend, divisor, multiplier, shift_right);
  }

  /// Computes integer division and modulus using precomputed values. This is
  /// computationally inexpensive.
  ///
  /// Simply returns the quotient
  __host__ __device__ __forceinline__ int divmod(int& remainder, int dividend)
      const {
    int quotient;
    fast_divmod(
        quotient, remainder, dividend, divisor, multiplier, shift_right);
    return quotient;
  }

  /// Computes integer division and modulus using precomputed values. This is
  /// computationally inexpensive.
  __host__ __device__ __forceinline__ void operator()(
      int& quotient,
      int64_t& remainder,
      int64_t dividend) const {
    fast_divmod(
        quotient, remainder, dividend, divisor, multiplier, shift_right);
  }

  /// Computes integer division and modulus using precomputed values. This is
  /// computationally inexpensive.
  __host__ __device__ __forceinline__ int divmod(
      int64_t& remainder,
      int64_t dividend) const {
    int quotient;
    fast_divmod(
        quotient, remainder, dividend, divisor, multiplier, shift_right);
    return quotient;
  }
};

} // namespace cu_ctc
