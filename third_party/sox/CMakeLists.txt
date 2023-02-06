include(FetchContent)

FetchContent_Declare(
  sox_src
  URL https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.bz2
  URL_HASH SHA256=81a6956d4330e75b5827316e44ae381e6f1e8928003c6aa45896da9041ea149c
  PATCH_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  )
# FetchContent_MakeAvailable will parse the downloaded content and setup the targets.
# We want to only download and not build, so we run Populate manually.
if(NOT sox_src_POPULATED)
  FetchContent_Populate(sox_src)
endif()

add_library(sox SHARED stub.c)
if(APPLE)
  set_target_properties(sox PROPERTIES SUFFIX .dylib)
endif(APPLE)
target_include_directories(sox PUBLIC ${sox_src_SOURCE_DIR}/src)
