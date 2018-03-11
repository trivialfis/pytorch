#ifndef AT_GENERIC_FILE
#define AT_GENERIC_FILE "ATen/glue/allocator.h"
#else

namespace at
{
#ifndef ATTHAllocator_dispatch
#define ATTHAllocator_dispatch
template <Backend B>
struct ATTHAllocator;
#endif

template <>
struct ATTHAllocator<Backend::Back>
{
  using type = TH_CONCAT_3(TH, Allocator_sym, Allocator);
};
} // namespace at

#endif
