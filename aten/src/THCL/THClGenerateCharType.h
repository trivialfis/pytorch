#ifndef THCL_GENERIC_FILE
#error "You must define THCL_GENERIC_FILE before including THGenerateCharType.h"
#endif

#define real int8_t
#define accreal int64_t
#define Real Char
#define ClReal ClChar
#define THCL_REAL_IS_CHAR
#line 1 THCL_GENERIC_FILE
#include THCL_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef ClReal
#undef THCL_REAL_IS_CHAR

#ifndef THClGenerateAllTypes
#undef THCL_GENERIC_FILE
#endif
