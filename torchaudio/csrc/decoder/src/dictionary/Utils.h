/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in
 * https://github.com/flashlight/flashlight/blob/d385b2150872fd7bf106601475d8719a703fe9ee/LICENSE
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "torchaudio/csrc/decoder/src/dictionary/Dictionary.h"

namespace torchaudio {
namespace lib {
namespace text {

using LexiconMap =
    std::unordered_map<std::string, std::vector<std::vector<std::string>>>;

Dictionary createWordDict(const LexiconMap& lexicon);

LexiconMap loadWords(const std::string& filename, int maxWords = -1);

// split word into tokens abc -> {"a", "b", "c"}
// Works with ASCII, UTF-8 encodings
std::vector<std::string> splitWrd(const std::string& word);

/**
 * Pack a token sequence by replacing consecutive repeats with replabels,
 * e.g. "abbccc" -> "ab1c2". The tokens "1", "2", ..., `to_string(maxReps)`
 * must already be in `dict`.
 */
std::vector<int> packReplabels(
    const std::vector<int>& tokens,
    const Dictionary& dict,
    int maxReps);

/**
 * Unpack a token sequence by replacing replabels with repeated tokens,
 * e.g. "ab1c2" -> "abbccc". The tokens "1", "2", ..., `to_string(maxReps)`
 * must already be in `dict`.
 */
std::vector<int> unpackReplabels(
    const std::vector<int>& tokens,
    const Dictionary& dict,
    int maxReps);
} // namespace text
} // namespace lib
} // namespace torchaudio
