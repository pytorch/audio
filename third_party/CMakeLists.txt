cmake_minimum_required(VERSION 3.1)

project(torchaudio_third_parties)
include(ExternalProject)

set(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install)
set(ARCHIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/archives)
set(COMMON_ARGS --quiet --disable-shared --enable-static --prefix=${INSTALL_DIR} --with-pic --disable-dependency-tracking --disable-debug --disable-examples --disable-doc)

# See the followings for the origin of patch
# http://www.linuxfromscratch.org/blfs/view/svn/multimedia/libmad.html
# http://www.linuxfromscratch.org/patches/blfs/svn/libmad-0.15.1b-fixes-1.patch
ExternalProject_Add(libmad
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://downloads.sourceforge.net/project/mad/libmad/0.15.1b/libmad-0.15.1b.tar.gz
  URL_HASH SHA256=bbfac3ed6bfbc2823d3775ebb931087371e142bb0e9bb1bee51a76a6e0078690
  PATCH_COMMAND patch < ${CMAKE_CURRENT_SOURCE_DIR}/libmad.patch
  CONFIGURE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/libmad/configure ${COMMON_ARGS}
)

ExternalProject_Add(libmp3lame
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://downloads.sourceforge.net/project/lame/lame/3.99/lame-3.99.5.tar.gz
  URL_HASH SHA256=24346b4158e4af3bd9f2e194bb23eb473c75fb7377011523353196b19b9a23ff
  CONFIGURE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/libmp3lame/configure ${COMMON_ARGS} --enable-nasm
)

ExternalProject_Add(libogg
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/ogg/libogg-1.3.3.tar.gz
  URL_HASH SHA256=c2e8a485110b97550f453226ec644ebac6cb29d1caef2902c007edab4308d985
  CONFIGURE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/libogg/configure ${COMMON_ARGS}
)

ExternalProject_Add(libflac
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS libogg
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz
  URL_HASH SHA256=91cfc3ed61dc40f47f050a109b08610667d73477af6ef36dcad31c31a4a8d53f
  CONFIGURE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/libflac/configure ${COMMON_ARGS} --with-ogg=${INSTALL_DIR}
)

ExternalProject_Add(libvorbis
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS libogg
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/vorbis/libvorbis-1.3.6.tar.gz
  URL_HASH SHA256=6ed40e0241089a42c48604dc00e362beee00036af2d8b3f46338031c9e0351cb
  CONFIGURE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/libvorbis/configure ${COMMON_ARGS} --with-ogg=${INSTALL_DIR}
)

ExternalProject_Add(libsox
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS libogg libflac libvorbis libmp3lame libmad
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.bz2
  URL_HASH SHA256=81a6956d4330e75b5827316e44ae381e6f1e8928003c6aa45896da9041ea149c
  CONFIGURE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/libsox/configure ${COMMON_ARGS} LDFLAGS=-L${INSTALL_DIR}/lib CPPFLAGS=-I${INSTALL_DIR}/include --with-lame --with-flac --with-mad --with-oggvorbis --without-alsa --without-coreaudio --without-png --without-oss --without-sndfile
)
