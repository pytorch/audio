// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once
#include <iostream>
#include <vector>
#include "include/ctc_prefix_decoder_host.h"

namespace cu_ctc {
constexpr size_t ALIGN_BYTES = 128;
constexpr int MAX_BLOCKS = 800;

template <typename T>
class DeviceDataWrap {
 public:
  DeviceDataWrap() : data_{}, size_in_bytes_{} {};
  DeviceDataWrap(T* data_ptr, size_t size_in_byte)
      : data_{data_ptr}, size_in_bytes_{size_in_byte} {};
  void print(size_t offset, size_t size_in_element, int eles_per_row = 10)
      const {
    if ((offset + size_in_element) * sizeof(T) > size_in_bytes_) {
      std::cerr
          << " ERROR: in DeviceDataWrap print : offset+size_in_element > size_in_bytes_";
      abort();
    }
    std::vector<T> host_data(size_in_element);
    CUDA_CHECK(cudaMemcpy(
        host_data.data(),
        data_ + offset,
        size_in_element * sizeof(T),
        cudaMemcpyDeviceToHost));

    for (int i = 0; i < size_in_element; ++i) {
      if (i != 0 && (i % eles_per_row == 0)) {
        std::cout << " \n";
      }
      std::cout << "[" << i << "]:" << host_data[i] << " ";
    }
    std::cout << "\n";
  }

  operator T*() {
    return data_;
  }

  operator const T*() {
    return const_cast<const T*>(data_);
  }

  T* data_ptr() const {
    return data_;
  }
  size_t size_in_byte() const {
    return size_in_bytes_;
  }
  void set_data_ptr(T* data_ptr) {
    data_ = data_ptr;
  }
  void set_size_in_byte(size_t size_in_byte) {
    size_in_bytes_ = size_in_byte;
  }

 private:
  T* data_;
  size_t size_in_bytes_;
};
} // namespace cu_ctc
