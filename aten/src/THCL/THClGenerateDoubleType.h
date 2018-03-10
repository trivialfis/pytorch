#ifndef THCL_GENERIC_FILE
#error "You must define THCL_GENERIC_FILE before including THGenerateDoubleType.h"
#endif

#define real double
#define accreal double
#define Real Double
#define CReal CudaDouble
#define THCL_REAL_IS_DOUBLE
#line 1 THCL_GENERIC_FILE
#include THCL_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THCL_REAL_IS_DOUBLE

#ifndef THClGenerateAllTypes
#ifndef THClGenerateFloatTypes
#undef THCL_GENERIC_FILE
#endif
#endif
