#ifndef THCL_GENERIC_FILE
#error "You must define THCL_GENERIC_FILE before including THGenerateFloatTypes.h"
#endif

#define THClGenerateFloatTypes

#define THClTypeIdxByte   1
#define THClTypeIdxChar   2
#define THClTypeIdxShort  3
#define THClTypeIdxInt    4
#define THClTypeIdxLong   5
#define THClTypeIdxFloat  6
#define THClTypeIdxDouble 7
#define THClTypeIdxHalf   8
#define THClTypeIdx_(T) TH_CONCAT_2(THClTypeIdx,T)

#include "THClGenerateHalfType.h"
#include "THClGenerateFloatType.h"
#include "THClGenerateDoubleType.h"

#undef THClTypeIdxByte
#undef THClTypeIdxChar
#undef THClTypeIdxShort
#undef THClTypeIdxInt
#undef THClTypeIdxLong
#undef THClTypeIdxFloat
#undef THClTypeIdxDouble
#undef THClTypeIdxHalf
#undef THClTypeIdx_

#undef THClGenerateFloatTypes
#undef THCL_GENERIC_FILE
