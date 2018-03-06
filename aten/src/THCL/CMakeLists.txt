set(ATen_CL_INCLUDE {ATen_CL_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  PARENT_SCOPE)

# CONFIGURE_FILE(THCLGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THCLGeneral.h")

# set(ATen_CL_SRCS ${ATen_CL_SRCS}
#   ${CMAKE_CURRENT_SOURCE_DIR}/THCLGeneral.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/THHalf.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/THAllocator.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/THSize.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/THStorage.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/THTensor.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/THBlas.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/THLapack.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/THLogAdd.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/THRandom.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/THFile.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/THDiskFile.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/THMemoryFile.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/THAtomic.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/THVector.cpp
#   ${extra_src}
# PARENT_SCOPE)

# INSTALL(FILES
#   TH.h
#   THAllocator.h
#   THMath.h
#   THBlas.h
#   THDiskFile.h
#   THFile.h
#   THFilePrivate.h
#   ${CMAKE_CURRENT_BINARY_DIR}/THCLGeneral.h
#   THGenerateAllTypes.h
#   THGenerateDoubleType.h
#   THGenerateFloatType.h
#   THGenerateHalfType.h
#   THGenerateLongType.h
#   THGenerateIntType.h
#   THGenerateShortType.h
#   THGenerateCharType.h
#   THGenerateByteType.h
#   THGenerateFloatTypes.h
#   THGenerateIntTypes.h
#   THLapack.h
#   THLogAdd.h
#   THMemoryFile.h
#   THRandom.h
#   THSize.h
#   THStorage.h
#   THTensor.h
#   THTensorApply.h
#   THTensorDimApply.h
#   THTensorMacros.h
#   THVector.h
#   THAtomic.h
#   THHalf.h
#   DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THCL")

# INSTALL(FILES
#   vector/AVX.h
#   vector/AVX2.h
#   vector/avx_mathfun.h
#   DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THCL/vector")

# INSTALL(FILES
#   generic/THBlas.c
#   generic/THBlas.h
#   generic/THLapack.cpp
#   generic/THLapack.h
#   generic/THStorage.c
#   generic/THStorage.h
#   generic/THStorageCopy.c
#   generic/THStorageCopy.h
#   generic/THTensor.cpp
#   generic/THTensor.h
#   generic/THTensorConv.cpp
#   generic/THTensorConv.h
#   generic/THTensorCopy.c
#   generic/THTensorCopy.h
#   generic/THTensorLapack.c
#   generic/THTensorLapack.h
#   generic/THTensorMath.c
#   generic/THTensorMath.h
#   generic/THTensorRandom.cpp
#   generic/THTensorRandom.h
#   generic/THVectorDispatch.cpp
#   generic/THVector.h
#   DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THCL/generic")
