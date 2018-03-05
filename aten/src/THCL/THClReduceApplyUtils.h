#ifndef THCL_REDUCE_APPLY_UTILS_INC
#define THCL_REDUCE_APPLY_UTILS_INC

#include <string>
#include <assert.h>
#include <stdexcept>

#include "THGeneral.h"
#include "THClGeneral.h"
#include "THClTensor.h"
#include "THClOperators.h"
#include "util/easycl_stringhelper.h"

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

// Maximum number of dimensions allowed for cltorch
#define MAX_CLTORCH_DIMS 25

// Warning string for tensor arguments that are too large or have too
// many dimensions
#define CLTORCH_STR(X) #X
#define CLTORCH_DIM_WARNING "tensor too large or too many (>" \
  CLTORCH_STR(MAX_CLTORCH_DIMS) ") dimensions"

std::string THClReduceApplyUtils_getKernelTemplate();

// Enum that indicates whether tensor arguments are read/write or
// read-only
enum TensorArgType { ReadWrite, ReadOnly };
class CLWrapper;

// Copy operator for the pointwise apply kernel
class CopyOp : public HasOperator2 {
public:
    std::string operator2() const {
        return "*out = *in1";
    }
};

// CL kernel argument that defines tensor layout
template <typename IndexType>
struct TensorInfo {
  // Extracts size/stride information for the kernel.
  // Successive dimensions can be collapsed if the size/strides match
  // up and thus there are no holes between the dimensions. This is used
  // to reduce the complexity of the problem.
  // The optional `reduceDim` indicates a reduction dimension for the
  // given tensor, so that the output size for this dimension will be 1.
  TensorInfo(THClState* state, THClTensor* t, int reduceDim = -1);

  // Contiguous tensors of more than one dimension are collapsed down
  // to one tensor
  inline bool isContiguous() const {
    return (dims == 1 && strides[0] == 1);
  }

  CLWrapper *wrapper;
  long offset;
//  float* data;
  IndexType sizes[MAX_CLTORCH_DIMS];
  IndexType strides[MAX_CLTORCH_DIMS];
  int dims;
};

template <typename IndexType>
TensorInfo<IndexType>::TensorInfo(THClState* state,
                                  THClTensor* t,
                                  int reduceDim)
    : wrapper(NULL), offset(0), dims(0) {
  int origDims = THClTensor_nDimension(state, t);
  assert(origDims <= MAX_CLTORCH_DIMS);
  assert(reduceDim < origDims);

  offset = THClTensor_storageOffset(state, t);
  wrapper = THClTensor_wrapper(state, t);

  // Count the number of successive dimensions that can be collapsed, from
  // innermost to outermost.
  int numCollapsed = 0;

  // Find the innermost dimension not of size 1, since dimensions of size 1 are
  // collapsible.
  int firstNonOneDim = -1;

  for (int i = origDims - 1; i >= 0; --i) {
    if (THClTensor_size(state, t, i) != 1 && i != reduceDim) {
      firstNonOneDim = i;
      break;
    }
  }

  // Special case: if all dimensions are of size 1, then this is a
  // single-point tensor that we still have to operate on. Reduce to a
  // single point.
  if (firstNonOneDim == -1) {
    dims = 1;
    sizes[0] = 1;
    strides[0] = 1;
    return;
  }

  // Skip the leading size 1 dims
  numCollapsed += origDims - 1 - firstNonOneDim;

  // Now, to determine the other collapsible dims. These are the size/strides
  // of the previous inner non-collapsible dim we encounter.
  long sizeInner = THClTensor_size(state, t, firstNonOneDim);
  IndexType strideInner = THClTensor_stride(state, t, firstNonOneDim);

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    long sizeOuter = (i == reduceDim) ? 1 : THClTensor_size(state, t, i);
    IndexType strideOuter = THClTensor_stride(state, t, i);

    // The next outermost dimension can be skipped if size 1
    if (sizeOuter == 1) {
      ++numCollapsed;
      continue;
    }

    // If the next outermost dimension is contiguous with the
    // previous non-collapsed one, collapse it
    if (strideOuter == strideInner * sizeInner) {
      ++numCollapsed;

      // This is the run of collapsed dimensions' size
      sizeInner = sizeInner * sizeOuter;
       continue;
    }

    // Otherwise, this new outer dimension at `i` cannot be collapsed
    // and is different from the previous.
    sizeInner = sizeOuter;
    strideInner = strideOuter;
  }

  assert(numCollapsed < origDims);
  dims = origDims - numCollapsed;

  // Determine the sizes of the collapsed dimensions.
  int collapsedIndex = origDims - numCollapsed - 1;
  sizes[collapsedIndex] = THClTensor_size(state, t, firstNonOneDim);
  strides[collapsedIndex] = THClTensor_stride(state, t, firstNonOneDim);

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    long sizeOuter = (i == reduceDim) ? 1 : THClTensor_size(state, t, i);
    IndexType strideOuter = THClTensor_stride(state, t, i);

    if (sizeOuter == 1) {
      // skip
      continue;
    }

    if (strideOuter == sizes[collapsedIndex] * strides[collapsedIndex]) {
      // collapse
      sizes[collapsedIndex] *= sizeOuter;
      continue;
    }

    // Otherwise, strides don't match; dimension `i` is not collapsible.
    --collapsedIndex;
    assert(collapsedIndex >= 0);
    sizes[collapsedIndex] = sizeOuter;
    strides[collapsedIndex] = strideOuter;
  }

  // We must have filled all the dimensions we're looking for
  assert(collapsedIndex == 0);
}


typedef struct TensorInfoCl {
  TensorInfoCl( TensorInfo<unsigned int> info ) {
    dims = info.dims;
    if( info.offset > ( 1l << 30 ) ) {
      throw std::runtime_error("size " + easycl::toString(info.offset) + " out of bounds");
    }
    offset = (int)info.offset;
    for( int i = 0; i < dims; i++ ) {
      sizes[i] = info.sizes[i];
      strides[i] = info.strides[i];
    }
  }
  TensorInfoCl( TensorInfo<unsigned long> info ) {
    dims = info.dims;
    if( info.offset > ( 1l << 30 ) ) {
      throw std::runtime_error("size " + easycl::toString(info.offset) + " out of bounds");
    }
    offset = (int)info.offset;
    for( int i = 0; i < dims; i++ ) {
      if( info.sizes[i] > ( 1l << 31 ) ) {
        throw std::runtime_error("size " + easycl::toString(info.sizes[i]) + " out of bounds");
      }
      sizes[i] = info.sizes[i];
      strides[i] = info.strides[i];
    }
  }
  TensorInfoCl( TensorInfo<unsigned long long> info ) {
    dims = info.dims;
    if( info.offset > ( 1l << 30 ) ) {
      throw std::runtime_error("size " + easycl::toString(info.offset) + " out of bounds");
    }
    offset = (int)info.offset;
    for( int i = 0; i < dims; i++ ) {
      if( info.sizes[i] > ( 1l << 31 ) ) {
        throw std::runtime_error("size " + easycl::toString(info.sizes[i]) + " out of bounds");
      }
      sizes[i] = info.sizes[i];
      strides[i] = info.strides[i];
    }
  }
  TensorInfoCl(THClTensor *tensor ) {
    dims = tensor->nDimension;
    for( int i = 0; i < dims; i++ ) {
      sizes[i] = tensor->size[i];
      strides[i] = tensor->stride[i];
    }
    offset = tensor->storageOffset;
  }
  unsigned int sizes[MAX_CLTORCH_DIMS];
  unsigned int strides[MAX_CLTORCH_DIMS];
  int offset;
  int dims;
} TensorInfoCl;


// Translate a linear index for the apply to a float* offset;
// specialized on `Dims` to reduce nvcc compilation time
//template <typename IndexType, int Dims>
//struct IndexToOffset {
//  static __host__ __device__ IndexType get(
//    IndexType linearId,
//    const TensorInfo<IndexType>& info) {
//    IndexType offset = 0;

//    // Use static dims
//    for (int i = Dims - 1; i >= 0; --i) {
//      IndexType curDimIndex = linearId % info.sizes[i];
//      IndexType curDimOffset = curDimIndex * info.strides[i];
//      offset += curDimOffset;

//      if (i > 0) {
//        linearId /= info.sizes[i];
//      }
//    }

//    return offset;
//  }
//};

//template <typename IndexType>
//struct IndexToOffset<IndexType, -2> {
//  static __forceinline__ __host__ __device__ IndexType
//    get(IndexType linearId, const TensorInfo<IndexType>& info) {
//    return linearId;
//  }
//};

//template <typename IndexType>
//struct IndexToOffset<IndexType, -1> {
//  static __forceinline__ __host__ __device__ IndexType
//    get(IndexType linearId, const TensorInfo<IndexType>& info) {
//    IndexType offset = 0;

//    // Use dynamic dims
//    for (int i = info.dims - 1; i >= 0; --i) {
//      IndexType curDimIndex = linearId % info.sizes[i];
//      IndexType curDimOffset = curDimIndex * info.strides[i];
//      offset += curDimOffset;

//      linearId /= info.sizes[i];
//    }

//    return offset;
//  }
//};

//template <typename IndexType>
//__device__ __forceinline__ IndexType getLinearBlockId() {
//  return blockIdx.z * gridDim.y * gridDim.x +
//    blockIdx.y * gridDim.x +
//    blockIdx.x;
//}

// Returns true if all linear ID -> offset math can be performed using 32 bit
// unsigned math, which is faster than 64 bit math
bool THCL_canUse32BitIndexMath(THClState* state, THClTensor* t);

// Produces a grid with at least one point per tile
bool THCL_getGridFromTiles(long gridTiles, dim3& grid);

// Determines if the given tensor has overlapping data points (i.e.,
// is there more than one index into the tensor that references the
// same piece of data)?
bool THCL_overlappingIndices(THClState* state, THClTensor* t);

#endif // THCL_REDUCE_APPLY_UTILS_INC
