include(ExternalProject)

set(INSTALL_DIR ${PROJECT_SOURCE_DIR}/install)
set(ARCHIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/archives)
set(COMMON_ARGS --quiet --disable-shared --enable-static --prefix=${INSTALL_DIR} --with-pic --disable-dependency-tracking --disable-debug --disable-examples --disable-doc)

# To pass custom environment variables to ExternalProject_Add command,
# we need to do `${CMAKE_COMMAND} -E env ${envs} <COMMANAD>`.
# https://stackoverflow.com/a/62437353
# We constrcut the custom environment variables here
set(envs
  "PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig"
  "LDFLAGS=-L${INSTALL_DIR}/lib $ENV{LDFLAGS}"
  "CPPFLAGS=-I${INSTALL_DIR}/include $ENV{CPPFLAGS}"
)

ExternalProject_Add(mad
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://downloads.sourceforge.net/project/mad/libmad/0.15.1b/libmad-0.15.1b.tar.gz
  URL_HASH SHA256=bbfac3ed6bfbc2823d3775ebb931087371e142bb0e9bb1bee51a76a6e0078690
  PATCH_COMMAND patch < ${CMAKE_CURRENT_SOURCE_DIR}/patch/libmad.patch
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/mad/configure ${COMMON_ARGS}
)

ExternalProject_Add(amr
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://sourceforge.net/projects/opencore-amr/files/opencore-amr/opencore-amr-0.1.5.tar.gz
  URL_HASH SHA256=2c006cb9d5f651bfb5e60156dbff6af3c9d35c7bbcc9015308c0aff1e14cd341
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/amr/configure ${COMMON_ARGS}
)

ExternalProject_Add(lame
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://downloads.sourceforge.net/project/lame/lame/3.99/lame-3.99.5.tar.gz
  URL_HASH SHA256=24346b4158e4af3bd9f2e194bb23eb473c75fb7377011523353196b19b9a23ff
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/lame/configure ${COMMON_ARGS} --enable-nasm
)

ExternalProject_Add(ogg
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/ogg/libogg-1.3.3.tar.gz
  URL_HASH SHA256=c2e8a485110b97550f453226ec644ebac6cb29d1caef2902c007edab4308d985
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/ogg/configure ${COMMON_ARGS}
)

ExternalProject_Add(flac
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ogg
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz
  URL_HASH SHA256=91cfc3ed61dc40f47f050a109b08610667d73477af6ef36dcad31c31a4a8d53f
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/flac/configure ${COMMON_ARGS} --with-ogg --disable-cpplibs
)

ExternalProject_Add(vorbis
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ogg
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/vorbis/libvorbis-1.3.6.tar.gz
  URL_HASH SHA256=6ed40e0241089a42c48604dc00e362beee00036af2d8b3f46338031c9e0351cb
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/vorbis/configure ${COMMON_ARGS} --with-ogg
)

ExternalProject_Add(opus
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ogg
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/opus/opus-1.3.1.tar.gz
  URL_HASH SHA256=65b58e1e25b2a114157014736a3d9dfeaad8d41be1c8179866f144a2fb44ff9d
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/opus/configure ${COMMON_ARGS} --with-ogg
)

ExternalProject_Add(opusfile
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS opus
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://ftp.osuosl.org/pub/xiph/releases/opus/opusfile-0.12.tar.gz
  URL_HASH SHA256=118d8601c12dd6a44f52423e68ca9083cc9f2bfe72da7a8c1acb22a80ae3550b
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/opusfile/configure ${COMMON_ARGS} --disable-http
)

ExternalProject_Add(sox
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ogg flac vorbis opusfile lame mad amr
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.bz2
  URL_HASH SHA256=81a6956d4330e75b5827316e44ae381e6f1e8928003c6aa45896da9041ea149c
  # OpenMP is by default compiled against GNU OpenMP, which conflicts with the version of OpenMP that PyTorch uses.
  # See https://github.com/pytorch/audio/pull/1026
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/sox/configure ${COMMON_ARGS} --with-lame --with-flac --with-mad --with-oggvorbis --without-alsa --without-coreaudio --without-png --without-oss --without-sndfile --with-opus --with-amrwb --with-amrnb --disable-openmp --without-sndio --without-pulseaudio
)
