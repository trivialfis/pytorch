#ifndef STORAGE_OPS
#define STORAGE_OPS

#include "ATen/ATenGeneral.h"
#include "ATen/ScalarType.h"
#include "ATen/Config.h"
#if AT_CL_ENABLED()
#include "THCL/THCl.h"
#endif
#include "ATen/StorageOpsGeneric.h"

#include "ATGenerate/GenerateAllBackendTypes.h"

#endif	// STORAGE_OPS
