/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in
 * https://github.com/flashlight/flashlight/blob/d385b2150872fd7bf106601475d8719a703fe9ee/LICENSE
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "torchaudio/csrc/decoder/src/decoder/LexiconDecoder.h"
#include "torchaudio/csrc/decoder/src/decoder/lm/KenLM.h"

namespace py = pybind11;
using namespace torchaudio::lib::text;
using namespace py::literals;

/**
 * Some hackery that lets pybind11 handle shared_ptr<void> (for old LMStatePtr).
 * See: https://github.com/pybind/pybind11/issues/820
 * PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
 * and inside PYBIND11_MODULE
 *   py::class_<std::shared_ptr<void>>(m, "encapsulated_data");
 */

namespace {

/**
 * A pybind11 "alias type" for abstract class LM, allowing one to subclass LM
 * with a custom LM defined purely in Python. For those who don't want to build
 * with KenLM, or have their own custom LM implementation.
 * See: https://pybind11.readthedocs.io/en/stable/advanced/classes.html
 *
 * TODO: ensure this works. Last time Jeff tried this there were slicing issues,
 * see https://github.com/pybind/pybind11/issues/1546 for workarounds.
 * This is low-pri since we assume most people can just build with KenLM.
 */
class PyLM : public LM {
  using LM::LM;

  // needed for pybind11 or else it won't compile
  using LMOutput = std::pair<LMStatePtr, float>;

  LMStatePtr start(bool startWithNothing) override {
    PYBIND11_OVERLOAD_PURE(LMStatePtr, LM, start, startWithNothing);
  }

  LMOutput score(const LMStatePtr& state, const int usrTokenIdx) override {
    PYBIND11_OVERLOAD_PURE(LMOutput, LM, score, state, usrTokenIdx);
  }

  LMOutput finish(const LMStatePtr& state) override {
    PYBIND11_OVERLOAD_PURE(LMOutput, LM, finish, state);
  }
};

/**
 * Using custom python LMState derived from LMState is not working with
 * custom python LM (derived from PyLM) because we need to to custing of LMState
 * in score and finish functions to the derived class
 * (for example vie obj.__class__ = CustomPyLMSTate) which cause the error
 * "TypeError: __class__ assignment: 'CustomPyLMState' deallocator differs
 * from 'flashlight.text.decoder._decoder.LMState'"
 * details see in https://github.com/pybind/pybind11/issues/1640
 * To define custom LM you can introduce map inside LM which maps LMstate to
 * additional state info (shared pointers pointing to the same underlying object
 * will have the same id in python in functions score and finish)
 *
 * ```python
 * from flashlight.lib.text.decoder import LM
 * class MyPyLM(LM):
 *      mapping_states = dict() # store simple additional int for each state
 *
 *      def __init__(self):
 *          LM.__init__(self)
 *
 *       def start(self, start_with_nothing):
 *          state = LMState()
 *          self.mapping_states[state] = 0
 *          return state
 *
 *      def score(self, state, index):
 *          outstate = state.child(index)
 *          if outstate not in self.mapping_states:
 *              self.mapping_states[outstate] = self.mapping_states[state] + 1
 *          return (outstate, -numpy.random.random())
 *
 *      def finish(self, state):
 *          outstate = state.child(-1)
 *          if outstate not in self.mapping_states:
 *              self.mapping_states[outstate] = self.mapping_states[state] + 1
 *          return (outstate, -1)
 *```
 */
void LexiconDecoder_decodeStep(
    LexiconDecoder& decoder,
    uintptr_t emissions,
    int T,
    int N) {
  decoder.decodeStep(reinterpret_cast<const float*>(emissions), T, N);
}

std::vector<DecodeResult> LexiconDecoder_decode(
    LexiconDecoder& decoder,
    uintptr_t emissions,
    int T,
    int N) {
  return decoder.decode(reinterpret_cast<const float*>(emissions), T, N);
}

} // namespace
