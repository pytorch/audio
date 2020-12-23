IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

project(rnnt_release)

IF (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
ENDIF()

IF (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    add_definitions(-DAPPLE)
ENDIF()

include_directories(submodule/include)

FIND_PACKAGE(CUDA)
MESSAGE(STATUS "cuda found ${CUDA_FOUND}")

option(USE_NAIVE_KERNEL "use naive alpha-beta kernel" OFF)
option(DEBUG_TIME "output kernel time" OFF)
option(DEBUG_KERNEL "output alpha beta" OFF)
if (USE_NAIVE_KERNEL)
    add_definitions(-DUSE_NAIVE_KERNEL)
endif()
if (DEBUG_TIME)
    add_definitions(-DDEBUG_TIME)
endif()
if (DEBUG_KERNEL)
    add_definitions(-DDEBUG_KERNEL)
endif()

option(WITH_GPU "compile warp-rnnt with cuda." ${CUDA_FOUND})
option(WITH_OMP "compile warp-rnnt with openmp." ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")

if(NOT WITH_OMP)
    add_definitions(-DRNNT_DISABLE_OMP)
endif()

if (WITH_OMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -fopenmp")
endif()

# need to be at least 30 or __shfl_down in reduce wont compile
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")
IF(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5)
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
ENDIF()

IF (CUDA_VERSION GREATER 7.6)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
ENDIF()

IF (CUDA_VERSION GREATER 8.9)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
ENDIF()

IF (CUDA_VERSION GREATER 9.9)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
ENDIF()

if (NOT APPLE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
ENDIF()


IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()


IF (WITH_GPU)

    MESSAGE(STATUS "Building static library with GPU support")

    CUDA_ADD_LIBRARY(warprnnt STATIC submodule/src/rnnt_entrypoint.cu)
    IF (!Torch_FOUND)
        TARGET_LINK_LIBRARIES(warprnnt ${CUDA_curand_LIBRARY})
    ENDIF()

    cuda_add_executable(test_time_gpu submodule/tests/test_time.cu submodule/tests/random.cpp )
    TARGET_LINK_LIBRARIES(test_time_gpu warprnnt ${CUDA_curand_LIBRARY})
    SET_TARGET_PROPERTIES(test_time_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    cuda_add_executable(test_gpu submodule/tests/test_gpu.cu submodule/tests/random.cpp )
    TARGET_LINK_LIBRARIES(test_gpu warprnnt ${CUDA_curand_LIBRARY})
    SET_TARGET_PROPERTIES(test_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

ELSE()
    MESSAGE(STATUS "Building static library with no GPU support")

    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    ENDIF()

    ADD_LIBRARY(warprnnt STATIC submodule/src/rnnt_entrypoint.cpp)

ENDIF()


add_executable(test_cpu submodule/tests/test_cpu.cpp submodule/tests/random.cpp )
TARGET_LINK_LIBRARIES(test_cpu warprnnt)
SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

add_executable(test_time submodule/tests/test_time.cpp submodule/tests/random.cpp )
TARGET_LINK_LIBRARIES(test_time warprnnt)
SET_TARGET_PROPERTIES(test_time PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

INSTALL(TARGETS warprnnt
        RUNTIME DESTINATION "bin"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib")

INSTALL(FILES submodule/include/rnnt.h DESTINATION "submodule/include")
