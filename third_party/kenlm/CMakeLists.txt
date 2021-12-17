set(
  KENLM_UTIL_SOURCES
  submodule/util/bit_packing.cc
  submodule/util/double-conversion/bignum.cc
  submodule/util/double-conversion/bignum-dtoa.cc
  submodule/util/double-conversion/cached-powers.cc
  submodule/util/double-conversion/diy-fp.cc
  submodule/util/double-conversion/double-conversion.cc
  submodule/util/double-conversion/fast-dtoa.cc
  submodule/util/double-conversion/fixed-dtoa.cc
  submodule/util/double-conversion/strtod.cc
  submodule/util/ersatz_progress.cc
  submodule/util/exception.cc
  submodule/util/file.cc
  submodule/util/file_piece.cc
  submodule/util/float_to_string.cc
  submodule/util/integer_to_string.cc
  submodule/util/mmap.cc
  submodule/util/murmur_hash.cc
  submodule/util/parallel_read.cc
  submodule/util/pool.cc
  submodule/util/read_compressed.cc
  submodule/util/scoped.cc
  submodule/util/spaces.cc
  submodule/util/stream/chain.cc
  submodule/util/stream/count_records.cc
  submodule/util/stream/io.cc
  submodule/util/stream/line_input.cc
  submodule/util/stream/multi_progress.cc
  submodule/util/stream/rewindable_stream.cc
  submodule/util/string_piece.cc
  submodule/util/usage.cc
  )

set(
  KENLM_SOURCES
  submodule/lm/bhiksha.cc
  submodule/lm/binary_format.cc
  submodule/lm/config.cc
  submodule/lm/lm_exception.cc
  submodule/lm/model.cc
  submodule/lm/quantize.cc
  submodule/lm/read_arpa.cc
  submodule/lm/search_hashed.cc
  submodule/lm/search_trie.cc
  submodule/lm/sizes.cc
  submodule/lm/trie.cc
  submodule/lm/trie_sort.cc
  submodule/lm/value_build.cc
  submodule/lm/virtual_interface.cc
  submodule/lm/vocab.cc
  )

add_library(
  kenlm
  STATIC
  "${KENLM_UTIL_SOURCES};${KENLM_SOURCES}"
  )

target_include_directories(
  kenlm
  BEFORE
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../install/include"
  PUBLIC submodule
  )

target_compile_definitions(
  kenlm
  PUBLIC KENLM_MAX_ORDER=6
  HAVE_ZLIB
  HAVE_BZLIB
  HAVE_XZLIB
  )

target_link_libraries(
  kenlm
  zlib
  bzip2
  lzma
)

add_dependencies(kenlm boost)
