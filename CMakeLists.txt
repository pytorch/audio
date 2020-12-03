cmake_minimum_required(VERSION 3.14)

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
  # Since this part of code is expected to be called from setup.py, we expect
  # that PYTHON_INCLUDE_DIR and PYTHON_LIBRARY are provided.
  # If needed, we can add search here, like
  # https://github.com/pytorch/pytorch/blob/7f869dca70606c42994d822ba11362a353411a1c/cmake/Dependencies.cmake#L904-L940
  message(STATUS "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")
  message(STATUS "PYTHON_LIBRARY: ${PYTHON_LIBRARY}")
  if (NOT PYTHON_INCLUDE_DIR)
    message(FATAL_ERROR "PYTHON_INCLUDE_DIR is not provided.")
  endif()
  if (NOT PYTHON_LIBRARY)
    message(FATAL_ERROR "PYTHON_LIBRARY is not provided.")
  endif()
  set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION})
  find_package(PythonInterp 3.6)
  find_package(PythonLibs 3.6 REQUIRED)
  message(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
  message(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")

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
    ${PYTHON_INCLUDE_DIRS}
    )

  # See https://github.com/pytorch/pytorch/issues/38122
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

  target_link_directories(
    _torchaudio
    )
  target_link_libraries(
    _torchaudio
    ${TORCHAUDIO_THIRD_PARTIES}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    ${PYTHON_LIBRARIES}
    )

  # We do not define install for _torchaudio.so
  # The location of installation is controlled by "CMAKE_LIBRARY_OUTPUT_DIRECTORY" and the
  # resulting file is handled by setuptools.
endif()
