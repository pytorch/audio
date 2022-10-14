if (NOT SOX_FOUND)
  find_package(PkgConfig)
  if (PKG_CONFIG_FOUND)
    pkg_check_modules(SOX sox)
    if (SOX_FOUND)
      message(STATUS "PkgConfig found sox (include: ${SOX_INCLUDE_DIRS}, link libraries: ${SOX_LINK_LIBRARIES})")
    endif()
  endif()

  if (NOT SOX_FOUND)
    set(SOX_ROOT $ENV{SOX_ROOT} CACHE PATH "Folder contains the sox library")

    find_path(SOX_INCLUDE_DIRS
      NAMES sox.h
      PATHS ${SOX_INCLUDE_DIR} ${SOX_ROOT}
      PATH_SUFFIXES "include" "include/sox/"
      )

    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      set(SOX_LIBNAME "libsox.dylib")
    else()
      SET(SOX_LIBNAME "libsox${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()

    find_library(SOX_LINK_LIBRARIES
      NAMES ${SOX_LIBNAME}
      PATHS ${SOX_LIB_DIR} ${SOX_ROOT}
      PATH_SUFFIXES "lib" "lib64"
      )

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(SOX DEFAULT_MSG SOX_INCLUDE_DIRS SOX_LINK_LIBRARIES)

    if(SOX_FOUND)
      set (SOX_HEADER_FILE "${SOX_INCLUDE_DIRS}/sox.h")
      message(STATUS "Found SOX (include: ${SOX_INCLUDE_DIRS}, library: ${SOX_LINK_LIBRARIES})")
      mark_as_advanced(SOX_ROOT SOX_INCLUDE_DIRS SOX_LINK_LIBRARIES)
    endif()
  endif()
endif()
