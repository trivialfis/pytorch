#ifndef THCL_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateHalfType.h"
#endif

#include "THClHalf.h"
#warning "What does it take to have proper half support?"
#define FORCE_TH_HALF 1
#if defined(CUDA_HALF_TENSOR) || defined(FORCE_TH_HALF)

#define real cl_half
#define accreal float
#define Real Half

// if only here via FORCE_TH_HALF, don't define ClReal since
// FORCE_TH_HALF should only be used for TH types
#ifdef CUDA_HALF_TENSOR
#define ClReal Half
#endif

#define THCL_REAL_IS_HALF
#line 1 THCL_GENERIC_FILE
#include THCL_GENERIC_FILE
#undef real
#undef accreal
#undef Real

#ifdef CUDA_HALF_TENSOR
#undef ClReal
#endif

#undef THCL_REAL_IS_HALF

#endif // defined(CUDA_HALF_TENSOR) || defined(FORCE_TH_HALF)

#ifndef THClGenerateAllTypes
#ifndef THClGenerateFloatTypes
#undef THCL_GENERIC_FILE
#endif
#endif
