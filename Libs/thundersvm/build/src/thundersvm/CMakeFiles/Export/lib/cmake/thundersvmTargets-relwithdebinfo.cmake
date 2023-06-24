#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "XComp::thundersvm" for configuration "RelWithDebInfo"
set_property(TARGET XComp::thundersvm APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(XComp::thundersvm PROPERTIES
  IMPORTED_IMPLIB_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/thundersvm.lib"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/bin/thundersvm.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS XComp::thundersvm )
list(APPEND _IMPORT_CHECK_FILES_FOR_XComp::thundersvm "${_IMPORT_PREFIX}/lib/thundersvm.lib" "${_IMPORT_PREFIX}/bin/thundersvm.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
