#ifndef THCL_STORAGE_INC
#define THCL_STORAGE_INC

#include "THStorage.h"
#include "THClGeneral.h"

#define THClStorage        TH_CONCAT_3(TH,CReal,Storage)
#define THClStorage_(NAME) TH_CONCAT_4(TH,CReal,Storage_,NAME)

/* fast access methods */
// #define THCL_STORAGE_GET(storage, idx) ((storage)->data[(idx)])
// #define THCL_STORAGE_SET(storage, idx, value) ((storage)->data[(idx)] = (value))

#include "generic/THClStorage.h"
#include "THClGenerateAllTypes.h"

#endif
