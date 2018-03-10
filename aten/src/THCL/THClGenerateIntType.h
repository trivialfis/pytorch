#ifndef THCL_GENERIC_FILE
#error "You must define THCL_GENERIC_FILE before including THGenerateIntType.h"
#endif

#define real int32_t
#define accreal int64_t
#define Real Int
#define CReal CudaInt
#define THCL_REAL_IS_INT
#line 1 THCL_GENERIC_FILE
#include THCL_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THCL_REAL_IS_INT

#ifndef THClGenerateAllTypes
#undef THCL_GENERIC_FILE
#endif
