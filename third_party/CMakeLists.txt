set(CMAKE_CXX_VISIBILITY_PRESET hidden)

file(MAKE_DIRECTORY install/include)
file(MAKE_DIRECTORY install/lib)

################################################################################
# sox
################################################################################
if (BUILD_SOX)
  add_subdirectory(sox)
endif()
