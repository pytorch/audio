include(ExternalProject)

set(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../install)
set(ARCHIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../archives)

# To pass custom environment variables to ExternalProject_Add command,
# we need to do `${CMAKE_COMMAND} -E env ${envs} <COMMANAD>`.
# https://stackoverflow.com/a/62437353
# We constrcut the custom environment variables here
set(envs
  "PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig"
  "LDFLAGS=-L${INSTALL_DIR}/lib $ENV{LDFLAGS}"
  "CFLAGS=-I${INSTALL_DIR}/include -fvisibility=hidden -fPIC $ENV{CFLAGS}"
)

set(
  LZMA_LIBRARIES
  ${INSTALL_DIR}/lib/liblzma.a
  )

ExternalProject_Add(lzma-
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://tukaani.org/xz/xz-5.2.5.tar.gz
  URL_HASH SHA256=f6f4910fd033078738bd82bfba4f49219d03b17eb0794eb91efbae419f4aba10
  BUILD_BYPRODUCTS ${LZMA_LIBRARIES}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/lzma-/configure --prefix=${INSTALL_DIR} --disable-xz --disable-xzdec --disable-lzmadec --disable-lzmainfo --disable-lzma-links --disable-scripts --disable-doc --enable-static --disable-shared
  BUILD_COMMAND  ${CMAKE_COMMAND} -E env ${envs} make VERBOSE=1
  DOWNLOAD_NO_PROGRESS ON
  LOG_DOWNLOAD ON
  LOG_UPDATE ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_MERGED_STDOUTERR ON
  LOG_OUTPUT_ON_FAILURE ON
)


add_library(lzma INTERFACE)
add_dependencies(lzma lzma-)
target_include_directories(lzma INTERFACE ${INSTALL_DIR}/include)
target_link_libraries(lzma INTERFACE ${LZMA_LIBRARIES})
