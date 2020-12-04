cmake_minimum_required(VERSION 3.14)

project(torchaudio)

option(BUILD_SOX "Build SoX and bind statically" OFF)
option(BUILD_LIBTORCHAUDIO "Build C++ Library" ON)
option(BUILD_PYTHON_EXTENSION "Build Python extension" OFF)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard whose features are requested to build this target.")

find_package(Torch REQUIRED)
# Set -D_GLIBCXX_USE_CXX11_ABI for third party builds
if (DEFINED _GLIBCXX_USE_CXX11_ABI)
  set(CXXFLAGS "${CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}")
endif()

add_subdirectory(third_party)

if (APPLE)
  SET(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

################################################################################
# Beginning of torchaudio extension configuration
################################################################################

set(TORCHAUDIO_CSRC "${CMAKE_CURRENT_SOURCE_DIR}/torchaudio/csrc")

set(
  LIBTORCHAUDIO_SOURCES
  "${TORCHAUDIO_CSRC}/sox_io.cpp"
  "${TORCHAUDIO_CSRC}/sox_utils.cpp"
  "${TORCHAUDIO_CSRC}/sox_effects.cpp"
  "${TORCHAUDIO_CSRC}/sox_effects_chain.cpp"
  "${TORCHAUDIO_CSRC}/register.cpp"
  )

################################################################################
# libtorchaudio.so
################################################################################
if(BUILD_LIBTORCHAUDIO)
  add_library(
    libtorchaudio
    SHARED
    ${LIBTORCHAUDIO_SOURCES}
    )
  set_target_properties(libtorchaudio PROPERTIES PREFIX "")

  target_include_directories(
    libtorchaudio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

  target_link_libraries(
    libtorchaudio
    "${TORCHAUDIO_THIRD_PARTIES}"
    "${TORCH_LIBRARIES}"
    )

  install(
    TARGETS
    libtorchaudio
    )

  set(TORCHAUDIO_LIBRARY -Wl,--no-as-needed libtorchaudio -Wl,--as-needed CACHE INTERNAL "")
endif()

################################################################################
# _torchaudio.so
################################################################################
if (BUILD_PYTHON_EXTENSION)
  find_package(Python REQUIRED Interpreter Development)

  add_library(
    _torchaudio
    SHARED
    ${TORCHAUDIO_CSRC}/sox.cpp
    ${LIBTORCHAUDIO_SOURCES}
    )

  set_target_properties(_torchaudio PROPERTIES PREFIX "")

  if (APPLE)
    # https://github.com/facebookarchive/caffe2/issues/854#issuecomment-364538485
    # https://github.com/pytorch/pytorch/commit/73f6715f4725a0723d8171d3131e09ac7abf0666
    set_target_properties(_torchaudio PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()

  target_include_directories(
    _torchaudio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python_INCLUDE_DIRS}
    )

  # See https://github.com/pytorch/pytorch/issues/38122
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

  target_link_libraries(
    _torchaudio
    ${TORCHAUDIO_THIRD_PARTIES}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    )

  # We do not define install for _torchaudio.so
  # The location of installation is controlled by "CMAKE_LIBRARY_OUTPUT_DIRECTORY" and the
  # resulting file is handled by setuptools.
endif()
