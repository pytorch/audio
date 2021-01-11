cmake_minimum_required(VERSION 3.5)

project(torchaudio_third_parties)

option(BUILD_SOX "Build libsox statically")
option(BUILD_TRANSDUCER "Build transducer statically")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (BUILD_SOX)
  add_subdirectory(sox)
endif()

if(BUILD_TRANSDUCER)
  add_subdirectory(transducer)
endif()
