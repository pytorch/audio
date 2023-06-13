set(
  sources
  io.cpp
  utils.cpp
  effects.cpp
  effects_chain.cpp
  types.cpp
  )
torchaudio_library(
  libtorchaudio_sox
  "${sources}"
  ""
  "torch;libsox"
  ""
  )

if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  torchaudio_extension(
    _torchaudio_sox
    "pybind/pybind.cpp;"
    ""
    "libtorchaudio_sox"
    ""
    )
endif()
