#ifndef THCL_GENERIC_FILE
#error "You must define THCL_GENERIC_FILE before including THGenerateLongType.h"
#endif

#define real int64_t
#define accreal int64_t
#define Real Long
#define ClReal ClLong
#define THCL_REAL_IS_LONG
#line 1 THCL_GENERIC_FILE
#include THCL_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef ClReal
#undef THCL_REAL_IS_LONG

#ifndef THClGenerateAllTypes
#undef THCL_GENERIC_FILE
#endif
