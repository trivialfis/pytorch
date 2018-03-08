#ifndef THCL_GENERIC_FILE
#error "You must define THCL_GENERIC_FILE before including THClGenerateAllTypes.h"
#endif

#ifndef THClGenerateManyTypes
#define THClAllLocalGenerateManyTypes
#define THClGenerateManyTypes
#endif

#include "THClGenerateFloatTypes.h"
#include "THClGenerateTypes.h"

#ifdef THAllLocalGenerateManyTypes
#undef THAllLocalGenerateManyTypes
#undef THClGenerateManyTypes
#undef THCL_GENERIC_FILE
#endif
