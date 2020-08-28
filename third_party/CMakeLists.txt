cmake_minimum_required(VERSION 3.1)

if (BUILD_SOX)

include(ExternalProject)

set(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install)
set(ARCHIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/archives)
set(COMMON_ARGS --quiet --disable-shared --enable-static --prefix=${INSTALL_DIR} --with-pic --disable-dependency-tracking --disable-debug --disable-examples --disable-doc)
set(envs "PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig" "LDFLAGS=-L${INSTALL_DIR}/lib $ENV{LDFLAGS}" "CPPFLAGS=-I${INSTALL_DIR}/include $ENV{CPPFLAGS}" "CXXFLAGS=${TORCH_CXX_FLAGS} $ENV{CXXFLAGS}")


ExternalProject_Add(mad_
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://downloads.sourceforge.net/project/mad/libmad/0.15.1b/libmad-0.15.1b.tar.gz
  URL_HASH SHA256=bbfac3ed6bfbc2823d3775ebb931087371e142bb0e9bb1bee51a76a6e0078690
  PATCH_COMMAND patch < ${CMAKE_CURRENT_SOURCE_DIR}/patch/libmad.patch
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_SOURCE_DIR}/src/mad_/configure ${COMMON_ARGS}
  BUILD_BYPRODUCTS ${INSTALL_DIR}/lib/libmad.a  
)

add_library(mad INTERFACE)
add_dependencies(mad mad_)
target_link_libraries(mad INTERFACE ${INSTALL_DIR}/lib/libmad.a)
target_include_directories(mad INTERFACE ${INSTALL_DIR}/include)

ExternalProject_Add(mp3lame_
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://downloads.sourceforge.net/project/lame/lame/3.99/lame-3.99.5.tar.gz
  URL_HASH SHA256=24346b4158e4af3bd9f2e194bb23eb473c75fb7377011523353196b19b9a23ff
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_SOURCE_DIR}/src/mp3lame_/configure ${COMMON_ARGS} --enable-nasm
  BUILD_BYPRODUCTS ${INSTALL_DIR}/lib/libmp3lame.a
)

add_library(mp3lame INTERFACE)
add_dependencies(mp3lame mp3lame_)
target_link_libraries(mp3lame INTERFACE ${INSTALL_DIR}/lib/libmp3lame.a)
target_include_directories(mp3lame INTERFACE ${INSTALL_DIR}/include)

ExternalProject_Add(ogg_
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/ogg/libogg-1.3.3.tar.gz
  URL_HASH SHA256=c2e8a485110b97550f453226ec644ebac6cb29d1caef2902c007edab4308d985
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_SOURCE_DIR}/src/ogg_/configure ${COMMON_ARGS}
  BUILD_BYPRODUCTS ${INSTALL_DIR}/lib/libogg.a
)

add_library(ogg INTERFACE)
add_dependencies(ogg ogg_)
target_link_libraries(ogg INTERFACE ${INSTALL_DIR}/lib/libogg.a)
target_include_directories(ogg INTERFACE ${INSTALL_DIR}/include)

ExternalProject_Add(flac_
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ogg_
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz
  URL_HASH SHA256=91cfc3ed61dc40f47f050a109b08610667d73477af6ef36dcad31c31a4a8d53f
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_SOURCE_DIR}/src/flac_/configure ${COMMON_ARGS} --with-ogg
  BUILD_BYPRODUCTS
  ${INSTALL_DIR}/lib/libFLAC.a
  ${INSTALL_DIR}/lib/libFLAC++.a
)

add_library(flac INTERFACE)
add_dependencies(flac flac_)
target_link_libraries(
  flac INTERFACE
  ${INSTALL_DIR}/lib/libFLAC.a
  ${INSTALL_DIR}/lib/libFLAC++.a
)
target_include_directories(flac INTERFACE ${INSTALL_DIR}/include)

ExternalProject_Add(vorbis_
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ogg_
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/vorbis/libvorbis-1.3.6.tar.gz
  URL_HASH SHA256=6ed40e0241089a42c48604dc00e362beee00036af2d8b3f46338031c9e0351cb
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_SOURCE_DIR}/src/vorbis_/configure ${COMMON_ARGS} --with-ogg
  BUILD_BYPRODUCTS
  ${INSTALL_DIR}/lib/libvorbisenc.a
  ${INSTALL_DIR}/lib/libvorbisfile.a
  ${INSTALL_DIR}/lib/libvorbis.a
)

add_library(vorbis INTERFACE)
add_dependencies(vorbis vorbis_)
target_link_libraries(
  vorbis INTERFACE
  ${INSTALL_DIR}/lib/libvorbisenc.a
  ${INSTALL_DIR}/lib/libvorbisfile.a
  ${INSTALL_DIR}/lib/libvorbis.a
)
target_include_directories(vorbis INTERFACE ${INSTALL_DIR}/include)

ExternalProject_Add(opus_
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ogg_
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/opus/opus-1.3.1.tar.gz
  URL_HASH SHA256=65b58e1e25b2a114157014736a3d9dfeaad8d41be1c8179866f144a2fb44ff9d
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_SOURCE_DIR}/src/opus_/configure ${COMMON_ARGS} --with-ogg
  BUILD_BYPRODUCTS ${INSTALL_DIR}/lib/libopus.a
)

add_library(opus INTERFACE)
add_dependencies(opus opus_)
target_link_libraries(opus INTERFACE ${INSTALL_DIR}/lib/libopus.a)
target_include_directories(opus INTERFACE ${INSTALL_DIR}/include)

ExternalProject_Add(opusfile_
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS opus_
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  STAMP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/opusfile-stamp
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/opusfile
  URL https://ftp.osuosl.org/pub/xiph/releases/opus/opusfile-0.12.tar.gz
  URL_HASH SHA256=118d8601c12dd6a44f52423e68ca9083cc9f2bfe72da7a8c1acb22a80ae3550b
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_SOURCE_DIR}/src/opusfile/configure ${COMMON_ARGS} --disable-http
  BUILD_BYPRODUCTS ${INSTALL_DIR}/lib/libopusfile.a
)

add_library(opusfile INTERFACE)
add_dependencies(opusfile opusfile_)
target_link_libraries(opusfile INTERFACE ${INSTALL_DIR}/lib/libopusfile.a)
target_include_directories(opusfile INTERFACE ${INSTALL_DIR}/include)

ExternalProject_Add(sox_
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ogg_ flac_ vorbis_ opusfile_ mp3lame_ mad_
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.bz2
  URL_HASH SHA256=81a6956d4330e75b5827316e44ae381e6f1e8928003c6aa45896da9041ea149c
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_SOURCE_DIR}/src/sox_/configure ${COMMON_ARGS} --with-lame --with-flac --with-mad --with-oggvorbis --without-alsa --without-coreaudio --without-png --without-oss --without-sndfile --with-opus --disable-openmp
  BUILD_BYPRODUCTS ${INSTALL_DIR}/lib/libsox.a
)

add_library(sox INTERFACE)
add_dependencies(sox sox_)
target_link_libraries(sox INTERFACE ${INSTALL_DIR}/lib/libsox.a)
target_include_directories(sox INTERFACE ${INSTALL_DIR}/include)

set(
  TORCHAUDIO_THIRD_PARTIES
  sox
  mad
  flac
  mp3lame
  opusfile
  opus
  vorbis
  ogg
  PARENT_SCOPE
)

else()
  set(TORCHAUDIO_THIRD_PARTIES sox)
endif()
