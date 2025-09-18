#pragma once
/*
  This header files provides CPP macros

    STABLE_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)

  that are torch::stable::Tensor-compatible analogous of
  the following macros:

    AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)

  respectively.

  TODO: remove this header file when torch::stable provides all
  features implemented here.
*/

#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/core/ScalarType.h>

namespace torchaudio::stable {

using torch::headeronly::ScalarType;

namespace impl {

template <ScalarType N>
struct ScalarTypeToCPPType;

#define SPECIALIZE_ScalarTypeToCPPType(cpp_type, scalar_type)   \
  template <>                                                \
  struct ScalarTypeToCPPType<ScalarType::scalar_type> { \
  using type = cpp_type;                                     \
  };

  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_ScalarTypeToCPPType)

#undef SPECIALIZE_ScalarTypeToCPPType

template <ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;
  
}  // namespace impl

}  // namespace torchaudio::stable
  
#define STABLE_DISPATCH_CASE(enum_type, ...) \
  case enum_type: {                                                           \
    using scalar_t [[maybe_unused]] = torchaudio::stable::impl::ScalarTypeToCPPTypeT<enum_type>; \
    return __VA_ARGS__();                                                     \
  }

#define STABLE_DISPATCH_SWITCH(TYPE, NAME, ...)                         \
  [&] {                                                                 \
    const auto& the_type = TYPE;                                        \
    constexpr const char* at_dispatch_name = NAME;                      \
    switch (the_type) {                                                 \
      __VA_ARGS__                                                       \
    default:                                                            \
      STD_TORCH_CHECK(                                                  \
            false,                                                          \
            '"',                                                            \
            at_dispatch_name,                                               \
            "\" not implemented for '",                                     \
            toString(the_type),                                                  \
            "'");                                                           \
    }                                                                       \
  }()

#define STABLE_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(...)   \
  STABLE_DISPATCH_CASE(ScalarType::Double, __VA_ARGS__)        \
  STABLE_DISPATCH_CASE(ScalarType::Float, __VA_ARGS__)  \
  STABLE_DISPATCH_CASE(ScalarType::Half, __VA_ARGS__)

#define STABLE_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
  STABLE_DISPATCH_SWITCH(                                        \
      TYPE, NAME, STABLE_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(__VA_ARGS__))
