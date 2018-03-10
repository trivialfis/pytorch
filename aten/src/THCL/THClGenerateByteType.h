#ifndef THCL_GENERIC_FILE
#error "You must define THCL_GENERIC_FILE before including THGenerateByteType.h"
#endif

#define real uint8_t
#define accreal int64_t
#define Real Byte
#define CReal CudaByte
#define THCL_REAL_IS_BYTE
#line 1 THCL_GENERIC_FILE
#include THCL_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THCL_REAL_IS_BYTE

#ifndef THClGenerateAllTypes
#undef THCL_GENERIC_FILE
#endif
