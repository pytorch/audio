cmake_minimum_required(VERSION 3.5)

project(libtorchaudio-cpp-example)

SET(BUILD_LIBTORCHAUDIO ON CACHE BOOL "Build libtorchaudio")
SET(BUILD_SOX ON CACHE BOOL "Build libsox into libtorchaudio")

SET(BUILD_KALDI OFF CACHE BOOL "Build Kaldi into libtorchaudio")
SET(BUILD_TRANSDUCER OFF CACHE BOOL "Build transducer into libtorchaudio")
SET(BUILD_TORCHAUDIO_PYTHON_EXTENSION OFF CACHE BOOL "Build Python binding")

find_package(Torch REQUIRED)
message("libtorchaudio CMakeLists: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(../.. libtorchaudio)
add_subdirectory(simplectc)
add_subdirectory(augmentation)
add_subdirectory(speech_recognition)
