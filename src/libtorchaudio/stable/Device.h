#pragma once

/*
  This header files provides torchaudio::stable::Device struct that is
  torch::stable::Tensor-compatible analogus of c10::Device defined
  c10/core/Device.h.

  TODO: remove this header file when torch::stable provides all
  features implemented here.
*/

#include <torch/csrc/stable/accelerator.h>

namespace torchaudio::stable {

using DeviceType = int32_t;
using torch::stable::accelerator::DeviceIndex;

struct Device {
  Device(DeviceType type, DeviceIndex index = -1) : type_(type), index_(index) {
    // TODO: validate();
  }

  /// Returns the type of device this is.
  DeviceType type() const noexcept {
    return type_;
  }

  /// Returns the optional index.
  DeviceIndex index() const noexcept {
    return index_;
  }

 private:
  DeviceType type_;
  DeviceIndex index_ = -1;
};

// A convinience function, not a part of torch::stable
inline Device cpu_device() {
  Device d(aoti_torch_device_type_cpu(), 0);
  return d;
}

} // namespace torchaudio::stable
