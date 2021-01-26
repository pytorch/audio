ADD_DEFINITIONS(-DRNNT_DISABLE_OMP)

IF(APPLE)
    ADD_DEFINITIONS(-DAPPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    # for el capitain have to use rpath
    IF(DARWIN_VERSION LESS 15)
        SET(CMAKE_SKIP_RPATH TRUE)
    ENDIF()

ELSE()
    # always skip for linux
    SET(CMAKE_SKIP_RPATH TRUE)
ENDIF()

ADD_LIBRARY(warprnnt STATIC submodule/src/rnnt_entrypoint.cpp)
target_include_directories(warprnnt PUBLIC submodule/include)
set_target_properties(warprnnt PROPERTIES PUBLIC_HEADER submodule/include/rnnt.h)


INSTALL(
  TARGETS warprnnt
  ARCHIVE DESTINATION "lib"
  PUBLIC_HEADER DESTINATION "include")
