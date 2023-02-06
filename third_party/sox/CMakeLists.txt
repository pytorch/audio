find_package(PkgConfig REQUIRED)

include(ExternalProject)

# set(INSTALL_DIR ${CMAKE_BINARY_DIR}/sox)
set(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../install)
set(ARCHIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../archives)
set(patch_dir ${PROJECT_SOURCE_DIR}/third_party/patches)
set(COMPILE_ARGS
  --quiet
  --enable-shared
  --disable-static
  --prefix=${INSTALL_DIR}
  --with-pic
  --disable-dependency-tracking
  --disable-debug
  --disable-examples
  --disable-doc
  --disable-openmp
  --without-amrnb
  --without-amrwb
  --without-flac
  --without-lame
  --without-oggvorbis
  --without-opus
  --without-alsa
  --without-ao
  --without-coreaudio
  --without-oss
  --without-id3tag
  --without-ladspa
  --without-mad
  --without-magic
  --without-png
  --without-pulseaudio
  --without-sndfile
  --without-sndio
  --without-sunaudio
  --without-waveaudio
  --without-wavpack
  --without-twolame
  )

if (APPLE)
  set(byproduct ${INSTALL_DIR}/lib/libsox.3.dylib)
  set(sox_library ${INSTALL_DIR}/lib/libsox.dylib)
else()
  set(byproduct ${INSTALL_DIR}/lib/libsox.so)
  set(sox_library ${INSTALL_DIR}/lib/libsox.so)
endif()

ExternalProject_Add(sox
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.bz2
  URL_HASH SHA256=81a6956d4330e75b5827316e44ae381e6f1e8928003c6aa45896da9041ea149c
  PATCH_COMMAND cp ${patch_dir}/config.guess ${patch_dir}/config.sub ${CMAKE_CURRENT_BINARY_DIR}/src/sox/
  CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/src/sox/configure ${COMPILE_ARGS}
  BUILD_BYPRODUCTS ${byproduct}
  DOWNLOAD_NO_PROGRESS ON
  LOG_DOWNLOAD ON
  LOG_UPDATE ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_MERGED_STDOUTERR ON
  LOG_OUTPUT_ON_FAILURE ON
)

if (APPLE)
  # Modify RPATH, so that they won't be hardcoded
  add_custom_command(
    OUTPUT ${sox_library}
    COMMAND install_name_tool -change ${byproduct} @rpath/libsox.dylib -id @rpath/libsox.dylib ${byproduct}
    DEPENDS sox
    )
  add_custom_target(_libsox DEPENDS ${sox_library})
else()
  add_custom_target(_libsox DEPENDS sox)
endif()

add_library(libsox INTERFACE)
add_dependencies(libsox _libsox)
target_include_directories(libsox INTERFACE ${INSTALL_DIR}/include)
target_link_libraries(libsox INTERFACE ${sox_library})
