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
  "prefix=${INSTALL_DIR}"
)

set(
  ZLIB_LIBRARIES
  ${INSTALL_DIR}/lib/libz.a
  )

ExternalProject_Add(zlib-
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://zlib.net/zlib-1.2.11.tar.gz
  URL_HASH SHA256=c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1
  BUILD_BYPRODUCTS ${ZLIB_LIBRARIES}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/zlib-/configure --static
  BUILD_COMMAND ${CMAKE_COMMAND} -E env ${envs} make VERBOSE=1
  LOG_DOWNLOAD ON
  LOG_UPDATE ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_MERGED_STDOUTERR ON
  LOG_OUTPUT_ON_FAILURE ON
)

add_library(zlib INTERFACE)
add_dependencies(zlib zlib-)
target_include_directories(zlib INTERFACE ${INSTALL_DIR}/include)
target_link_libraries(zlib INTERFACE ${ZLIB_LIBRARIES})
