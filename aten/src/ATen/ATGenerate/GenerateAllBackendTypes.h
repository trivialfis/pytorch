#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#include "ATen/Config.h"

#ifndef THGenerateManyTypes
#define THAllLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

// Backend in enum `Backend`
#define Back CPU
// Backend symbol used by low level TH libraries
#define Back_sym
// #include "ATen/ATGenerate/GenerateAllTypes.h"
#include "TH/THGenerateAllTypes.h"
#undef Back_sym
#undef Back

#if AT_CL_ENABLED()
#define Back CL
#define Back_sym Cl
// #include "ATen/ATGenerate/GenerateAllTypes.h"
#include "THCL/THClGenerateAllTypes.h"
#undef Back_sym
#undef Back
#endif

#if AT_CUDA_ENABLED()
#define Back CUDA
#define Back_sym C
// #include "ATen/ATGenerate/GenerateAllTypes.h"
#include "THC/THCGenerateAllTypes.h"
#undef Back_sym
#undef Back
#endif

#ifdef THAllLocalGenerateManyTypes
#undef THAllLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
