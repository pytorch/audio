include(ExternalProject)

set(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../install)
set(ARCHIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../archives)

if (MSVC)
  set(
    ZLIB_LIBRARIES
    ${INSTALL_DIR}/lib/zlib.lib
  )

  ExternalProject_Add(zlib-
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}
    DOWNLOAD_DIR ${ARCHIVE_DIR}
    URL https://github.com/madler/zlib/archive/refs/tags/v1.2.12.tar.gz
    URL_HASH SHA256=d8688496ea40fb61787500e863cc63c9afcbc524468cedeb478068924eb54932
    BUILD_BYPRODUCTS ${ZLIB_LIBRARIES}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND nmake -f win32/Makefile.msc zlib.lib
    BUILD_COMMAND cp zlib.lib ${INSTALL_DIR}/lib/
    INSTALL_COMMAND cp zlib.h zconf.h ${INSTALL_DIR}/include/
    LOG_DOWNLOAD ON
    LOG_UPDATE ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    LOG_MERGED_STDOUTERR ON
    LOG_OUTPUT_ON_FAILURE ON
  )
else()
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
    URL https://github.com/madler/zlib/archive/refs/tags/v1.2.12.tar.gz
    URL_HASH SHA256=d8688496ea40fb61787500e863cc63c9afcbc524468cedeb478068924eb54932
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
endif()

add_library(zlib INTERFACE)
add_dependencies(zlib zlib-)
target_include_directories(zlib INTERFACE ${INSTALL_DIR}/include)
target_link_libraries(zlib INTERFACE ${ZLIB_LIBRARIES})
