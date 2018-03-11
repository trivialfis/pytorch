#ifndef THCL_GENERIC_FILE
#error "You must define THCL_GENERIC_FILE before including THGenerateFloatType.h"
#endif

#define real cl_half
/* FIXME: fp64 has bad performance on some platforms; avoid using it unless
   we opt into it? */
#define accreal float
#define Real Half
#define ClReal ClHalf
#define THCL_REAL_IS_HALFx
#line 1 THCL_GENERIC_FILE
#include THCL_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef ClReal
#undef THCL_REAL_IS_FLOAT

#ifndef THClGenerateAllTypes
#ifndef THClGenerateFloatTypes
#undef THCL_GENERIC_FILE
#endif
#endif
