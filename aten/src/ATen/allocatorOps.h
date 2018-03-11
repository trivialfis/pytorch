#ifndef ALLOCATOR_OPS
#define ALLOCATOR_OPS

#include "ATen/ATenGeneral.h"
#include "ATen/ScalarType.h"
#include "ATen/Config.h"
#include "ATen/glue/allocator.h"

#include "TH/TH.h"
// struct THAllocator;
// #if AT_CL_ENABLED()
#include "THCL/THCl.h"
// struct THClDeviceAllocator;
// #endif

#define Back CPU
#define Allocator_sym
#define AT_GENERIC_FILE "ATen/glue/allocator.h"
#line 1 AT_GENERIC_FILE
#include AT_GENERIC_FILE
#undef Allocator_sym
#undef Back

#if AT_CL_ENABLED()
#define Back CL
#define Allocator_sym ClDevice
#line 1 AT_GENERIC_FILE
#include AT_GENERIC_FILE
#undef Allocator_sym
#undef Back
#endif	// AT_CL_ENABLED

#undef AT_GENERIC_FILE

#endif
