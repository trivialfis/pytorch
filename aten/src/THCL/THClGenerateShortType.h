#ifndef THCL_GENERIC_FILE
#error "You must define THCL_GENERIC_FILE before including THGenerateShortType.h"
#endif

#define real int16_t
#define accreal int64_t
#define Real Short
#define ClReal ClShort
#define THCL_REAL_IS_SHORT
#line 1 THCL_GENERIC_FILE
#include THCL_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef ClReal
#undef THCL_REAL_IS_SHORT

#ifndef THClGenerateAllTypes
#undef THCL_GENERIC_FILE
#endif
