#pragma once

#include <torch/csrc/stable/tensor.h>
#include <type_traits>
#include <cstdarg>

using torch::stable::Tensor;

template<unsigned int k, typename T, bool IsConst = true>
class Accessor {
  int64_t strides[k];
  T *data;

public:
  using tensor_type = typename std::conditional<IsConst, const Tensor&, Tensor&>::type;

  Accessor(tensor_type tensor) {
    auto raw_ptr = tensor.data_ptr();
    data = static_cast<T*>(raw_ptr);
    for (unsigned int i = 0; i < k; i++) {
      strides[i] = tensor.stride(i);
    }
  }

  T index(...) {
    va_list args;
    va_start(args, k);
    int64_t ix = 0;
    for (unsigned int i = 0; i < k; i++) {
        ix += strides[i] * va_arg(args, int);
    }
    va_end(args);
    return data[ix];
  }

  template<bool C = IsConst>
  typename std::enable_if<!C, void>::type set_index(T value, ...) {
    va_list args;
    va_start(args, value);
    int64_t ix = 0;
    for (unsigned int i = 0; i < k; i++) {
        ix += strides[i] * va_arg(args, int);
    }
    va_end(args);
    data[ix] = value;
  }
};
