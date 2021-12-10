/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in
 * https://github.com/flashlight/flashlight/blob/d385b2150872fd7bf106601475d8719a703fe9ee/LICENSE
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "torchaudio/csrc/decoder/src/dictionary/Dictionary.h"
#include "torchaudio/csrc/decoder/src/dictionary/Utils.h"

namespace py = pybind11;
using namespace torchaudio::lib::text;
using namespace py::literals;

namespace {

void Dictionary_addEntry_0(
    Dictionary& dict,
    const std::string& entry,
    int idx) {
  dict.addEntry(entry, idx);
}

void Dictionary_addEntry_1(Dictionary& dict, const std::string& entry) {
  dict.addEntry(entry);
}

} // namespace
