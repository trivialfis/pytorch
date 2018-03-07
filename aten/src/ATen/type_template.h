#include "ATen/ATenGeneral.h"
#include "ATen/ArrayRef.h"
#include "ATen/Generator.h"
#include "ATen/Half.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/ScalarType.h"
#include "ATen/Scalar.h"
// #include "ATen/Tensor.h"
#include "ATen/Allocator.h"
#include "ATen/Type.h"

namespace at{
template <Backend b, ScalarType s>
struct dummytype final : public Type
{
  explicit dummytype(Context* context)
    : Type(context) {}
};
#define DEFINE_CL_TYPE(_1, n, _2)					\
  using CL ## n ## Type = dummytype<Backend::CL, ScalarType::n>;
#define DEFINE_CPU_TYPE(_1, n, _2)					\
  using CPU ## n ## Type = dummytype<Backend::CPU, ScalarType::n>;
#define DEFINE_CUDA_TYPE(_1, n, _2)					\
  using CUDA ## n ## Type = dummytype<Backend::CUDA, ScalarType::n>
}
