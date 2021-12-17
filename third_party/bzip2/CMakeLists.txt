include(ExternalProject)

set(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../install)
set(ARCHIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../archives)

ExternalProject_Add(bzip2_
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
  URL_HASH SHA256=ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND make install PREFIX=${INSTALL_DIR}
  DOWNLOAD_NO_PROGRESS ON
  LOG_DOWNLOAD ON
  LOG_UPDATE ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_MERGED_STDOUTERR ON
  LOG_OUTPUT_ON_FAILURE ON
)

add_library(bzip2 INTERFACE)
add_dependencies(bzip2 bzip2_)
target_include_directories(bzip2 INTERFACE ${INSTALL_DIR}/include)
target_link_libraries(bzip2 INTERFACE ${INSTALL_DIR}/lib/libbz2.a)
