#include <torch/extension.h>

#include <torchaudio/csrc/decoder/bindings/_decoder.cpp>
#include <torchaudio/csrc/decoder/bindings/_dictionary.cpp>

PYBIND11_MODULE(_torchaudio_decoder, m) {
#ifdef BUILD_CTC_DECODER
  py::enum_<SmearingMode>(m, "_SmearingMode")
      .value("NONE", SmearingMode::NONE)
      .value("MAX", SmearingMode::MAX)
      .value("LOGADD", SmearingMode::LOGADD);

  py::class_<TrieNode, TrieNodePtr>(m, "_TrieNode")
      .def(py::init<int>(), "idx"_a)
      .def_readwrite("children", &TrieNode::children)
      .def_readwrite("idx", &TrieNode::idx)
      .def_readwrite("labels", &TrieNode::labels)
      .def_readwrite("scores", &TrieNode::scores)
      .def_readwrite("max_score", &TrieNode::maxScore);

  py::class_<Trie, TriePtr>(m, "_Trie")
      .def(py::init<int, int>(), "max_children"_a, "root_idx"_a)
      .def("get_root", &Trie::getRoot)
      .def("insert", &Trie::insert, "indices"_a, "label"_a, "score"_a)
      .def("search", &Trie::search, "indices"_a)
      .def("smear", &Trie::smear, "smear_mode"_a);

  py::class_<LM, LMPtr, PyLM>(m, "_LM")
      .def(py::init<>())
      .def("start", &LM::start, "start_with_nothing"_a)
      .def("score", &LM::score, "state"_a, "usr_token_idx"_a)
      .def("finish", &LM::finish, "state"_a);

  py::class_<LMState, LMStatePtr>(m, "_LMState")
      .def(py::init<>())
      .def_readwrite("children", &LMState::children)
      .def("compare", &LMState::compare, "state"_a)
      .def("child", &LMState::child<LMState>, "usr_index"_a);

  py::class_<KenLM, KenLMPtr, LM>(m, "_KenLM")
      .def(
          py::init<const std::string&, const Dictionary&>(),
          "path"_a,
          "usr_token_dict"_a);

  py::enum_<CriterionType>(m, "_CriterionType")
      .value("ASG", CriterionType::ASG)
      .value("CTC", CriterionType::CTC);

  py::class_<LexiconDecoderOptions>(m, "_LexiconDecoderOptions")
      .def(
          py::init<
              const int,
              const int,
              const double,
              const double,
              const double,
              const double,
              const double,
              const bool,
              const CriterionType>(),
          "beam_size"_a,
          "beam_size_token"_a,
          "beam_threshold"_a,
          "lm_weight"_a,
          "word_score"_a,
          "unk_score"_a,
          "sil_score"_a,
          "log_add"_a,
          "criterion_type"_a)
      .def_readwrite("beam_size", &LexiconDecoderOptions::beamSize)
      .def_readwrite("beam_size_token", &LexiconDecoderOptions::beamSizeToken)
      .def_readwrite("beam_threshold", &LexiconDecoderOptions::beamThreshold)
      .def_readwrite("lm_weight", &LexiconDecoderOptions::lmWeight)
      .def_readwrite("word_score", &LexiconDecoderOptions::wordScore)
      .def_readwrite("unk_score", &LexiconDecoderOptions::unkScore)
      .def_readwrite("sil_score", &LexiconDecoderOptions::silScore)
      .def_readwrite("log_add", &LexiconDecoderOptions::logAdd)
      .def_readwrite("criterion_type", &LexiconDecoderOptions::criterionType);

  py::class_<DecodeResult>(m, "_DecodeResult")
      .def(py::init<int>(), "length"_a)
      .def_readwrite("score", &DecodeResult::score)
      .def_readwrite("amScore", &DecodeResult::amScore)
      .def_readwrite("lmScore", &DecodeResult::lmScore)
      .def_readwrite("words", &DecodeResult::words)
      .def_readwrite("tokens", &DecodeResult::tokens);

  // NB: `decode` and `decodeStep` expect raw emissions pointers.
  py::class_<LexiconDecoder>(m, "_LexiconDecoder")
      .def(py::init<
           LexiconDecoderOptions,
           const TriePtr,
           const LMPtr,
           const int,
           const int,
           const int,
           const std::vector<float>&,
           const bool>())
      .def("decode_begin", &LexiconDecoder::decodeBegin)
      .def(
          "decode_step",
          &LexiconDecoder_decodeStep,
          "emissions"_a,
          "T"_a,
          "N"_a)
      .def("decode_end", &LexiconDecoder::decodeEnd)
      .def("decode", &LexiconDecoder_decode, "emissions"_a, "T"_a, "N"_a)
      .def("prune", &LexiconDecoder::prune, "look_back"_a = 0)
      .def(
          "get_best_hypothesis",
          &LexiconDecoder::getBestHypothesis,
          "look_back"_a = 0)
      .def("get_all_final_hypothesis", &LexiconDecoder::getAllFinalHypothesis);

  py::class_<Dictionary>(m, "_Dictionary")
      .def(py::init<>())
      .def(py::init<const std::string&>(), "filename"_a)
      .def("entry_size", &Dictionary::entrySize)
      .def("index_size", &Dictionary::indexSize)
      .def("add_entry", &Dictionary_addEntry_0, "entry"_a, "idx"_a)
      .def("add_entry", &Dictionary_addEntry_1, "entry"_a)
      .def("get_entry", &Dictionary::getEntry, "idx"_a)
      .def("set_default_index", &Dictionary::setDefaultIndex, "idx"_a)
      .def("get_index", &Dictionary::getIndex, "entry"_a)
      .def("contains", &Dictionary::contains, "entry"_a)
      .def("is_contiguous", &Dictionary::isContiguous)
      .def(
          "map_entries_to_indices",
          &Dictionary::mapEntriesToIndices,
          "entries"_a)
      .def(
          "map_indices_to_entries",
          &Dictionary::mapIndicesToEntries,
          "indices"_a);
  m.def("_create_word_dict", &createWordDict, "lexicon"_a);
  m.def("_load_words", &loadWords, "filename"_a, "max_words"_a = -1);
#endif
}
