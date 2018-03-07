#include "ATen/ATenGeneral.h"
#include "ATen/ArrayRef.h"
#include "ATen/Generator.h"
#include "ATen/Half.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/ScalarType.h"
#include "ATen/Scalar.h"
// #include "ATen/Tensor.h"
#include "ATen/Allocator.h"

namespace at{
template <Backend b, ScalarType s>
AT_API struct dummytype
{
};
}
