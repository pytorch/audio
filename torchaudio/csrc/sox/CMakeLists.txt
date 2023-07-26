set(
  sources
  stub.cpp
  io.cpp
  utils.cpp
  effects.cpp
  effects_chain.cpp
  types.cpp
  )
set(
  compiler_definitions
  DLOPEN_SOX
  )

torchaudio_library(
  libtorchaudio_sox
  "${sources}"
  ""
  "torch;libsox"
  "${compiler_definitions}"
  )

if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  torchaudio_extension(
    _torchaudio_sox
    "pybind/pybind.cpp;"
    ""
    "libtorchaudio_sox"
    "${compiler_definitions}"
    )
endif()
