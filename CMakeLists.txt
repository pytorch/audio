cmake_minimum_required(VERSION 3.18)

project(torchaudio)

option(BUILD_SOX "Build SoX and bind statically" OFF)
option(BUILD_LIBTORCHAUDIO "Build C++ Library" ON)
option(BUILD_PYTHON_EXTENSION "Build Python extension" OFF)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard whose features are requested to build this target.")

find_package(Torch REQUIRED)
# TORCH_CXX_FLAGS contains proper _GLIBCXX_USE_CXX11_ABI
# We propagate it to CMAKE_CXX_FLAGS so that third party libraries are built
# with the same flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(third_party)

################################################################################
# Begining of torchaudio extension configuration
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
  # Note:
  # It seems that there is a bug in find_package(Python COMPONENTS Development) and
  # the resulting Python_LIBRARIES point a directory instead of the library
  # file, and it causes the link step to fail.
  # So, until find_package is fixed (and we change the cmake_minimum_required
  # to that specific version), we request Python_LIBRARIES and Python_INCLUDE_DIRS
  # to be manually specified.
  # This approach does not require an extra step from a user, because we do so in
  # setup.py
  find_package(Python3 REQUIRED COMPONENTS Development.Module)

  if (NOT Python3_LIBRARIES)
    message(FATAL_ERROR "Python_LIBRARIES is not defined. Try with -DPython_LIBRARIES=...")
  endif()
  if (NOT Python3_INCLUDE_DIRS)
    message(FATAL_ERROR "Python_include_dirs is not defined. Try with -DPython_INCLUDE_DIR=...?")
  endif()

  add_library(
    _torchaudio
    SHARED
    ${TORCHAUDIO_CSRC}/sox.cpp
    ${LIBTORCHAUDIO_SOURCES}
    )

  set_target_properties(_torchaudio PROPERTIES PREFIX "")

  target_include_directories(
    _torchaudio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
    )

  # See https://github.com/pytorch/pytorch/issues/38122
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

  target_link_libraries(
    _torchaudio
    ${TORCHAUDIO_THIRD_PARTIES}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    ${Python3_LIBRARIES}
    )

  # We do not define install for _torchaudio.so
  # The location of installation is controlled by "CMAKE_LIBRARY_OUTPUT_DIRECTORY" and the
  # resulting file is handled by setuptools.
endif()
