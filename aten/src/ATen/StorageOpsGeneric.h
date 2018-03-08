#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "ATen/StorageOpsGeneric.h"
#else
#include "ATen/ScalarType.h"
namespace at
{

#define AT_OPS(return_type, name, arguments, ...)		\
  template <Backend B, ScalarType S>				\
  struct name ## _op;						\
  template <>							\
  struct name ## _op<Backend::Back, ScalarType::Real>		\
  {								\
    static return_type name arguments				\
      __VA_ARGS__;						\
  };

#define TH_CONCAT_5_EXPAND(a, b, c, d, e) a ## b ## c ## d ## e
#define TH_CONCAT_5(a, b, c, d, e)   TH_CONCAT_5_EXPAND(a, b, c, d, e)

#define ATStorage                TH_CONCAT_4(TH, Back_sym, Real, Storage)
#define ATStorage_(NAME)         TH_CONCAT_5(TH, Back_sym, Real, Storage_, NAME)

AT_OPS(ATStorage*, _new, (),
       {
	 return ATStorage_(new)();
       })
AT_OPS(ATStorage*, _new_with_size, (long size)
       {
	 return ATStorage_(newWithSize)(size);
       });
AT_OPS(ATStorage*, _new_with_allocator, (std::size_t size,
					 THAllocator* allocator,
					 void *allocatorContext),
       {
	 return ATStorage_(newWithAllocator)(size, allocator, allocatorContext);
       })
AT_OPS(ATStorage*, _new_with_data_and_allocator,
       (real* data, long size, THAllocator* allocator, void* allocatorContext),
       {
	 return ATStorage_(newWithDataAndAllocator)(data,
						    size,
						    allocator,
						    allocatorContext);
       })
AT_OPS(void, _free, (ATStorage *storage),
       {
	 ATStorage_(free)(storage);
       })
AT_OPS(void, _retain, (ATStorage *storage),
       {
	 ATStorage_(retain)(storage);
       })
AT_OPS(void, _resize, (ATStorage *storage, int64_t new_size),
       {
	 ATStorage_(resize)(storage, new_size);
       })
AT_OPS(void, _fill, (ATStorage* storage, int value),
       {
	 ATStorage_(fill)(storage, value);
       })
AT_OPS(void, _set, (ATStorage* storage, size_t ind, int value),
       {
	 ATStorage_(set)(storage, ind, value);
       })
AT_OPS(real, _get, (ATStorage* storage, std::size_t ind),
       {
	 return ((ATStorage_(get)(storage, ind)));
       })
AT_OPS(void, _set_flag, (ATStorage* storage, char flag),
       {
	 ATStorage_(setFlag)(storage, flag);
       })
AT_OPS(void, _clear_flag, (ATStorage* storage, char flag),
       {
	 ATStorage_(clearFlag)(storage, flag);
       })

} // namespace at

#endif
