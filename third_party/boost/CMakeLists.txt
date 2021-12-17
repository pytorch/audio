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
  "CFLAGS=-I${INSTALL_DIR}/include -fvisibility=hidden $ENV{CFLAGS}"
)

set(
  BOOST_LIBRARIES
  ${INSTALL_DIR}/lib/libboost_thread.a
  )

ExternalProject_Add(boost_
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz
  URL_HASH SHA256=94ced8b72956591c4775ae2207a9763d3600b30d9d7446562c552f0a14a63be7
  BUILD_BYPRODUCTS ${BOOST_LIBRARIES}
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ./bootstrap.sh --with-libraries=thread --prefix=${INSTALL_DIR}
  BUILD_COMMAND ${CMAKE_COMMAND} -E env ${envs} ./b2 link=static
  INSTALL_COMMAND ${CMAKE_COMMAND} -E env ${envs} ./b2 link=static install
  DOWNLOAD_NO_PROGRESS ON
  LOG_DOWNLOAD ON
  LOG_UPDATE ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_MERGED_STDOUTERR ON
  LOG_OUTPUT_ON_FAILURE ON
)

add_library(boost INTERFACE)
add_dependencies(boost boost_)
target_include_directories(boost INTERFACE ${INSTALL_DIR}/include)
target_link_libraries(boost INTERFACE ${BOOST_LIBRARIES})
