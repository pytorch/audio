/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * https://github.com/flashlight/flashlight/blob/d385b2150872fd7bf106601475d8719a703fe9ee/LICENSE
 */

#pragma once

#include "torchaudio/csrc/decoder/src/decoder/lm/LM.h"

namespace torchaudio {
namespace lib {
namespace text {

/**
 * ZeroLM is a dummy language model class, which mimics the behavior of a
 * uni-gram language model but always returns 0 as score.
 */
class ZeroLM : public LM {
 public:
  LMStatePtr start(bool startWithNothing) override;

  std::pair<LMStatePtr, float> score(
      const LMStatePtr& state,
      const int usrTokenIdx) override;

  std::pair<LMStatePtr, float> finish(const LMStatePtr& state) override;
};

using ZeroLMPtr = std::shared_ptr<ZeroLM>;
} // namespace text
} // namespace lib
} // namespace torchaudio
