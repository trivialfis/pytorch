#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "ATen/StorageOpsGeneric.h"
#else

#include <ATen/ScalarType.h>
/*
  Anything use of macro technique for code generation should stop at this layer.
*/
namespace at
{
// Generate storage types.
#ifndef ATTHStorage_dispatch
#define ATTHStorage_dispatch
template <Backend B, ScalarType S>
struct ATTHStorage;
#endif

template <>
struct ATTHStorage<Backend::Back, ScalarType::Real>
{
  using type = TH_CONCAT_4(TH, Back_sym, Real, Storage);
};

#define TH_CONCAT_5_EXPAND(a, b, c, d, e)             a ## b ## c ## d ## e
#define TH_CONCAT_5(a, b, c, d, e)        TH_CONCAT_5_EXPAND(a, b, c, d, e)

#define ATStorage_(NAME)    TH_CONCAT_5(TH, Back_sym, Real, Storage_, NAME)
#define AT_STORAGE_TYPE typename ATTHStorage<B, S>::type
#define AT_STORAGE_DISPATCH typename ATTHStorage<Backend::Back, ScalarType::Real>::type

#define CASOE(name)					\
  template <Backend B, ScalarType S>			\
  struct name ;						\

#ifndef ATTHStorage_new_op
#define ATTHStorage_new_op
CASOE(ATTHStorage_new)
#endif
template <>
struct ATTHStorage_new<Backend::Back, ScalarType::Real>
{
  static AT_STORAGE_DISPATCH* op ()
  {
    return ATStorage_(new)();
  }
};

#ifndef ATTHStorage_new_with_size_op
#define ATTHStorage_new_with_size_op
CASOE(ATTHStorage_new_with_size)
#endif
template <>
struct ATTHStorage_new_with_size<Backend::Back, ScalarType::Real>
{
  static AT_STORAGE_DISPATCH* op (ptrdiff_t size)
  {
    return ATStorage_(newWithSize)(size);
  };
};

#ifndef ATTHStorage_new_with_allocator_op
#define ATTHStorage_new_with_allocator_op
CASOE(ATTHStorage_new_with_allocator)
#endif
template <>
struct ATTHStorage_new_with_allocator<Backend::Back, ScalarType::Real>
{
  static AT_STORAGE_DISPATCH* op(ptrdiff_t size,
				 THAllocator* allocator,
				 void *allocatorContext)
  {
    return ATStorage_(newWithAllocator)(size, allocator, allocatorContext);
  }
};

#ifndef ATTHStorage_new_with_data_and_allocator_op
#define ATTHStorage_new_with_data_and_allocator_op
CASOE(ATTHStorage_new_with_data_and_allocator)
#endif
template <>
struct ATTHStorage_new_with_data_and_allocator<Backend::Back, ScalarType::Real>
{
  static AT_STORAGE_DISPATCH* op(real* data, ptrdiff_t size, THAllocator* allocator,
				 void* allocatorContext)
  {
    return ATStorage_(newWithDataAndAllocator)(data, size, allocator,
					       allocatorContext);
  }
};

#ifndef ATTHStorage_free_op
#define ATTHStorage_free_op
CASOE(ATTHStorage_free)
#endif
template <>
struct ATTHStorage_free<Backend::Back, ScalarType::Real>
{
  static void op(AT_STORAGE_DISPATCH *storage)
  {
    ATStorage_(free)(storage);
  }
};

#ifndef ATTHStorage_retain_op
#define ATTHStorage_retain_op
CASOE(ATTHStorage_retain)
#endif
template <>
struct ATTHStorage_retain<Backend::Back, ScalarType::Real>
{
  static void op(AT_STORAGE_DISPATCH *storage)
  {
    ATStorage_(retain)(storage);
  }
};

#ifndef ATTHStorage_resize_op
#define ATTHStorage_resize_op
CASOE(ATTHStorage_resize)
#endif
template<>
struct ATTHStorage_resize<Backend::Back, ScalarType::Real>
{
  static void op(AT_STORAGE_DISPATCH *storage, ptrdiff_t new_size)
  {
    ATStorage_(resize)(storage, new_size);
  }
};

#ifndef ATTHStorage_fill_op
#define ATTHStorage_fill_op
CASOE(ATTHStorage_fill)
#endif
template<>
struct ATTHStorage_fill<Backend::Back, ScalarType::Real>
{
  static void op(AT_STORAGE_DISPATCH *storage, real value)
  {
    ATStorage_(fill)(storage, value);
  }
};

#ifndef ATTHStorage_set_op
#define ATTHStorage_set_op
CASOE(ATTHStorage_set)
#endif
template <>
struct ATTHStorage_set<Backend::Back, ScalarType::Real>
{
  static void op(AT_STORAGE_DISPATCH *storage, ptrdiff_t ind, real value)
  {
    ATStorage_(set)(storage, ind, value);
  }
};

#ifndef ATTHStorage_get_op
#define ATTHStorage_get_op
CASOE(ATTHStorage_get)
#endif
template <>
struct ATTHStorage_get<Backend::Back, ScalarType::Real>
{
  static real op(AT_STORAGE_DISPATCH *storage, ptrdiff_t ind)
  {
    return ((ATStorage_(get)(storage, ind)));
  }
};

#ifndef ATTHStorage_set_flag_op
#define  ATTHStorage_set_flag_op
CASOE(ATTHStorage_set_flag)
#endif
template <>
struct ATTHStorage_set_flag<Backend::Back, ScalarType::Real>
{
  static void op(AT_STORAGE_DISPATCH *storage, char flag)
  {
    ATStorage_(setFlag)(storage, flag);
  }
};


#ifndef ATTHStorage_clear_flag_op
#define ATTHStorage_clear_flag_op
CASOE(ATTHStorage_clear_flag)
#endif
template <>
struct ATTHStorage_clear_flag<Backend::Back, ScalarType::Real>
{
  static void op(AT_STORAGE_DISPATCH *storage, char flag)
  {
    ATStorage_(clearFlag)(storage, flag);
  }
};
} // namespace at

#endif	// TH_GENERIC_FILE
