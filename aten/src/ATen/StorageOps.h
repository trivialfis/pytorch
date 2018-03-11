#ifndef STORAGE_OPS
#define STORAGE_OPS

#include "ATen/ATenGeneral.h"
#include "ATen/ScalarType.h"
#include "ATen/Config.h"
// #include "TH/THGeneral.h"

#define TH_CONCAT_3(x,y,z) TH_CONCAT_3_EXPAND(x,y,z)
#define TH_CONCAT_3_EXPAND(x,y,z)        x ## y ## z

#include "ATen/allocatorOps.h"

#define Back CPU
#define Back_sym
#define Allocator_sym
#define TH_GENERIC_FILE "ATen/glue/StorageOpsGeneric.h"
// #include TH_GENERIC_FILE
#include "TH/THGenerateAllTypes.h"
#undef Allocator_sym
#undef Back_sym
#undef Back

#undef TH_GENERIC_FILE

#if AT_CL_ENABLED()
#define Back CL
#define Back_sym Cl
#define Allocator_sym ClDevice
#define THCL_GENERIC_FILE "ATen/glue/StorageOpsGeneric.h"
#include "THCL/THClGenerateAllTypes.h"
#warning "CL Generated." 
#undef Allocator_sym
#undef Back_sym
#undef Back

#undef THCL_GENERIC_FILE

namespace at{
ATTHStorage<Backend::CL, ScalarType::Byte> test_temp_byte;
ATTHStorage<Backend::CL, ScalarType::Float> test_temp_float;
ATTHStorage<Backend::CL, ScalarType::Double> test_temp_double;
ATTHStorage<Backend::CL, ScalarType::Half> test_temp_half;
}

#endif

#if AT_CUDA_ENABLED()
#define Back CUDA
#define Back_sym CReal
#define Allocator_sym CDevice
#define THC_GENERIC_FILE "ATen/glue/StorageOpsGeneric.h"
#include "THC/THCGenerateAllTypes.h"
#undef Allocator_sym
#undef Back_sym
#undef Back
#endif

#undef THC_GENERIC_FILE

#endif	// STORAGE_OPS
