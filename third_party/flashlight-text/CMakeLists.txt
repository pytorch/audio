# Custom CMakeLists for building flashlight-text decoder
#
# The main difference from upstream native CMakeLists from flashlight-text.
#
# 1. Build compression libraries statically and make KenLM self-contained
# 2. Build KenLM without Boost by compiling only what is used by flashlight-text
# 3. Build KenLM and flashlight-text in one go (not required, but nice-to-have feature)
# 4. Tweak the location of bindings so that its easier for TorchAudio build process to pick up.
#    (the upstream CMakeLists.txt does not install them in the same location as libflashlight-text)
# 5. Tweak the name of bindings. (remove suffix like cpython-37m-darwin)

set(CMAKE_CXX_VISIBILITY_PRESET default)

# the following line is added in order to export symbols when building on Windows
# this approach has some limitations as documented in https://github.com/pytorch/pytorch/pull/3650
if (MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

set(
  libflashlight_src
  submodule/flashlight/lib/text/decoder/Utils.cpp
  submodule/flashlight/lib/text/decoder/lm/KenLM.cpp
  submodule/flashlight/lib/text/decoder/lm/ZeroLM.cpp
  submodule/flashlight/lib/text/decoder/lm/ConvLM.cpp
  submodule/flashlight/lib/text/decoder/LexiconDecoder.cpp
  submodule/flashlight/lib/text/decoder/LexiconFreeDecoder.cpp
  submodule/flashlight/lib/text/decoder/LexiconFreeSeq2SeqDecoder.cpp
  submodule/flashlight/lib/text/decoder/LexiconSeq2SeqDecoder.cpp
  submodule/flashlight/lib/text/decoder/Trie.cpp
  submodule/flashlight/lib/text/String.cpp
  submodule/flashlight/lib/text/dictionary/Utils.cpp
  submodule/flashlight/lib/text/dictionary/Dictionary.cpp
  )

torchaudio_library(
  libflashlight-text
  "${libflashlight_src}"
  submodule
  ""
  FL_TEXT_USE_KENLM
  )

# TODO: update torchaudio_library to handle private links
target_link_libraries(
  libflashlight-text
  PRIVATE
  kenlm)

if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  torchaudio_extension(
    flashlight_lib_text_dictionary
    submodule/bindings/python/flashlight/lib/text/_dictionary.cpp
    submodule
    libflashlight-text
    ""
    )
  torchaudio_extension(
    flashlight_lib_text_decoder
    submodule/bindings/python/flashlight/lib/text/_decoder.cpp
    submodule
    libflashlight-text
    FL_TEXT_USE_KENLM
    )
endif()
