/**
 *  Modified from Rapidsai/raft(https://github.com/rapidsai/raft)
 *
 */

#pragma once
#include <type_traits>

namespace cu_ctc {
/**
 * @brief Give logarithm of the number to base-2
 * @tparam IntType data type (checked only for integers)
 */
template <typename IntType>
constexpr __host__ __device__ IntType
log2(IntType num, IntType ret = IntType(0)) {
  return num <= IntType(1) ? ret : log2(num >> IntType(1), ++ret);
}

/**
 * @brief Fast arithmetics and alignment checks for power-of-two values known at
 * compile time.
 *
 * @tparam Value_ a compile-time value representable as a power-of-two.
 */
template <auto Value_>
struct Pow2 {
  using Type = decltype(Value_);
  static constexpr Type Value = Value_;
  static constexpr Type Log2 = log2(Value);
  static constexpr Type Mask = Value - 1;

  static_assert(std::is_integral<Type>::value, "Value must be integral.");
  static_assert(Value && !(Value & Mask), "Value must be power of two.");

#define Pow2_FUNC_QUALIFIER static constexpr __host__ __device__ __forceinline__
#define Pow2_WHEN_INTEGRAL(I) std::enable_if_t<Pow2_IS_REPRESENTABLE_AS(I), I>
#define Pow2_IS_REPRESENTABLE_AS(I) \
  (std::is_integral<I>::value && Type(I(Value)) == Value)

  /**
   * Integer division by Value truncated toward zero
   * (same as `x / Value` in C++).
   *
   *  Invariant: `x = Value * quot(x) + rem(x)`
   */
  template <typename I>
  Pow2_FUNC_QUALIFIER Pow2_WHEN_INTEGRAL(I) quot(I x) noexcept {
    if constexpr (std::is_signed<I>::value)
      return (x >> I(Log2)) + (x < 0 && (x & I(Mask)));
    if constexpr (std::is_unsigned<I>::value)
      return x >> I(Log2);
  }

  /**
   *  Remainder of integer division by Value truncated toward zero
   *  (same as `x % Value` in C++).
   *
   *  Invariant: `x = Value * quot(x) + rem(x)`.
   */
  template <typename I>
  Pow2_FUNC_QUALIFIER Pow2_WHEN_INTEGRAL(I) rem(I x) noexcept {
    if constexpr (std::is_signed<I>::value)
      return x < 0 ? -((-x) & I(Mask)) : (x & I(Mask));
    if constexpr (std::is_unsigned<I>::value)
      return x & I(Mask);
  }

  /**
   * Integer division by Value truncated toward negative infinity
   * (same as `x // Value` in Python).
   *
   * Invariant: `x = Value * div(x) + mod(x)`.
   *
   * Note, `div` and `mod` for negative values are slightly faster
   * than `quot` and `rem`, but behave slightly different
   * compared to normal C++ operators `/` and `%`.
   */
  template <typename I>
  Pow2_FUNC_QUALIFIER Pow2_WHEN_INTEGRAL(I) div(I x) noexcept {
    return x >> I(Log2);
  }

  /**
   * x modulo Value operation (remainder of the `div(x)`)
   * (same as `x % Value` in Python).
   *
   * Invariant: `mod(x) >= 0`
   * Invariant: `x = Value * div(x) + mod(x)`.
   *
   * Note, `div` and `mod` for negative values are slightly faster
   * than `quot` and `rem`, but behave slightly different
   * compared to normal C++ operators `/` and `%`.
   */
  template <typename I>
  Pow2_FUNC_QUALIFIER Pow2_WHEN_INTEGRAL(I) mod(I x) noexcept {
    return x & I(Mask);
  }

#define Pow2_CHECK_TYPE(T)                                     \
  static_assert(                                               \
      std::is_pointer<T>::value || std::is_integral<T>::value, \
      "Only pointer or integral types make sense here")

  /**
   * Tell whether the pointer or integral is Value-aligned.
   * NB: for pointers, the alignment is checked in bytes, not in elements.
   */
  template <typename PtrT>
  Pow2_FUNC_QUALIFIER bool isAligned(PtrT p) noexcept {
    Pow2_CHECK_TYPE(PtrT);
    if constexpr (Pow2_IS_REPRESENTABLE_AS(PtrT))
      return mod(p) == 0;
    if constexpr (!Pow2_IS_REPRESENTABLE_AS(PtrT))
      return mod(reinterpret_cast<Type>(p)) == 0;
  }

  /** Tell whether two pointers have the same address modulo Value. */
  template <typename PtrT, typename PtrS>
  Pow2_FUNC_QUALIFIER bool areSameAlignOffsets(PtrT a, PtrS b) noexcept {
    Pow2_CHECK_TYPE(PtrT);
    Pow2_CHECK_TYPE(PtrS);
    Type x, y;
    if constexpr (Pow2_IS_REPRESENTABLE_AS(PtrT))
      x = Type(mod(a));
    else
      x = mod(reinterpret_cast<Type>(a));
    if constexpr (Pow2_IS_REPRESENTABLE_AS(PtrS))
      y = Type(mod(b));
    else
      y = mod(reinterpret_cast<Type>(b));
    return x == y;
  }

  /** Get this or next Value-aligned address (in bytes) or integral. */
  template <typename PtrT>
  Pow2_FUNC_QUALIFIER PtrT roundUp(PtrT p) noexcept {
    Pow2_CHECK_TYPE(PtrT);
    if constexpr (Pow2_IS_REPRESENTABLE_AS(PtrT))
      return (p + PtrT(Mask)) & PtrT(~Mask);
    if constexpr (!Pow2_IS_REPRESENTABLE_AS(PtrT)) {
      auto x = reinterpret_cast<Type>(p);
      return reinterpret_cast<PtrT>((x + Mask) & (~Mask));
    }
  }

  /** Get this or previous Value-aligned address (in bytes) or integral. */
  template <typename PtrT>
  Pow2_FUNC_QUALIFIER PtrT roundDown(PtrT p) noexcept {
    Pow2_CHECK_TYPE(PtrT);
    if constexpr (Pow2_IS_REPRESENTABLE_AS(PtrT))
      return p & PtrT(~Mask);
    if constexpr (!Pow2_IS_REPRESENTABLE_AS(PtrT)) {
      auto x = reinterpret_cast<Type>(p);
      return reinterpret_cast<PtrT>(x & (~Mask));
    }
  }
#undef Pow2_CHECK_TYPE
#undef Pow2_IS_REPRESENTABLE_AS
#undef Pow2_FUNC_QUALIFIER
#undef Pow2_WHEN_INTEGRAL
};

}; // namespace cu_ctc
