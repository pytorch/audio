set(
  KENLM_UTIL_SOURCES
  kenlm/util/bit_packing.cc
  kenlm/util/double-conversion/bignum.cc
  kenlm/util/double-conversion/bignum-dtoa.cc
  kenlm/util/double-conversion/cached-powers.cc
  kenlm/util/double-conversion/diy-fp.cc
  kenlm/util/double-conversion/double-conversion.cc
  kenlm/util/double-conversion/fast-dtoa.cc
  kenlm/util/double-conversion/fixed-dtoa.cc
  kenlm/util/double-conversion/strtod.cc
  kenlm/util/ersatz_progress.cc
  kenlm/util/exception.cc
  kenlm/util/file.cc
  kenlm/util/file_piece.cc
  kenlm/util/float_to_string.cc
  kenlm/util/integer_to_string.cc
  kenlm/util/mmap.cc
  kenlm/util/murmur_hash.cc
  kenlm/util/pool.cc
  kenlm/util/read_compressed.cc
  kenlm/util/scoped.cc
  kenlm/util/spaces.cc
  kenlm/util/string_piece.cc
  )

set(
  KENLM_SOURCES
  kenlm/lm/bhiksha.cc
  kenlm/lm/binary_format.cc
  kenlm/lm/config.cc
  kenlm/lm/lm_exception.cc
  kenlm/lm/model.cc
  kenlm/lm/quantize.cc
  kenlm/lm/read_arpa.cc
  kenlm/lm/search_hashed.cc
  kenlm/lm/search_trie.cc
  kenlm/lm/trie.cc
  kenlm/lm/trie_sort.cc
  kenlm/lm/value_build.cc
  kenlm/lm/virtual_interface.cc
  kenlm/lm/vocab.cc
  )

add_library(
  kenlm
  STATIC
  "${KENLM_UTIL_SOURCES};${KENLM_SOURCES}"
  )

target_include_directories(
  kenlm
  BEFORE
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
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
  PUBLIC
  zlib
  bzip2
  lzma
)
