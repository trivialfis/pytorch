#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "ATen/StorageOpsGeneric.h"
#else

namespace at
{

// Generate storage types.
template <Backend B, ScalarType S>
struct ATTHStorage;

#define CREATE_AT_STORAGES()					\
  template <>							\
  struct ATTHStorage<Backend::Back, ScalarType::Real>		\
  {								\
    using type = TH_CONCAT_4(TH, Back_sym, Real, Storage);	\
  };

CREATE_AT_STORAGES()

#define TH_CONCAT_5_EXPAND(a, b, c, d, e)             a ## b ## c ## d ## e
#define TH_CONCAT_5(a, b, c, d, e)        TH_CONCAT_5_EXPAND(a, b, c, d, e)

#define ATStorage_(NAME)    TH_CONCAT_5(TH, Back_sym, Real, Storage_, NAME)

// Things are geting tedious really fast.
#define AT_TEMPLATE template <Backend B, ScalarType S>
#define AT_STORAGE_TYPE typename ATTHStorage<B, S>::type

AT_TEMPLATE
AT_STORAGE_TYPE* ATTHStorage_new()
{
  return ATStorage_(new)();
}

AT_TEMPLATE
AT_STORAGE_TYPE* ATTHStorage_new_with_size(long size)
{
  return ATStorage_(newWithSize)(size);
}

AT_TEMPLATE
AT_STORAGE_TYPE* ATTHStorage_new_with_allocator(std::size_t size,
						 THAllocator* allocator,
						 void *allocatorContext)
{
  return ATStorage_(newWithAllocator)(size, allocator, allocatorContext);
}

AT_TEMPLATE
AT_STORAGE_TYPE*
ATTHStorage_new_with_data_and_allocator(real* data,
					long size,
					THAllocator* allocator,
					void* allocatorContext)
{
  return ATStorage_(newWithDataAndAllocator)(data,
					     size,
					     allocator,
					     allocatorContext);
}

AT_TEMPLATE
void ATTHStorage_free(AT_STORAGE_TYPE *storage)
{
  ATStorage_(free)(storage);
}

AT_TEMPLATE
void ATTHStorage_retain(AT_STORAGE_TYPE *storage)
{
  ATStorage_(retain)(storage);
}

AT_TEMPLATE
void ATTHStorage_resize(AT_STORAGE_TYPE *storage, int64_t new_size)
{
  ATStorage_(resize)(storage, new_size);
}

AT_TEMPLATE
void ATTHStorage_fill(AT_STORAGE_TYPE *storage, int value)
{
  ATStorage_(fill)(storage, value);
}

AT_TEMPLATE
void ATTHStorage_set(AT_STORAGE_TYPE *storage, size_t ind, int value)
{
  ATStorage_(set)(storage, ind, value);
}

AT_TEMPLATE
real ATTHStorage_get(AT_STORAGE_TYPE *storage, std::size_t ind)
{
  return ((ATStorage_(get)(storage, ind)));  
}

AT_TEMPLATE
void ATTHStorage_set_flag(AT_STORAGE_TYPE *storage, char flag)
{
  ATStorage_(setFlag)(storage, flag);
}

AT_TEMPLATE
void ATTHStorage_clear_flag(AT_STORAGE_TYPE *storage, char flag)
{
  ATStorage_(clearFlag)(storage, flag);  
}
} // namespace at

#endif
