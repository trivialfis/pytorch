// #include "ATen/StorageTemplate.h"

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "ATen/StorageOpsGeneric.h"
#else
#include "ATen/ScalarType.h"
namespace at
{

#define AT_OPS(return_type, name, backend, arguments, ...)	\
  template <Backend B, ScalarType S>				\
  struct name ## _op;						\
  template <>							\
  struct name ## _op<backend, ScalarType::Real>			\
  {								\
    static return_type name arguments				\
      __VA_ARGS__;						\
  };

AT_OPS(THStorage*, _new, Backend::CPU, (),
       {
	 return THStorage_(new)();
       })
AT_OPS(THStorage*, _new_with_size, Backend::CPU, (long size)
       {
	 return THStorage_(newWithSize)(size);
       });
AT_OPS(THStorage*, _new_with_allocator, Backend::CPU, (std::size_t size, THAllocator* allocator, void *allocatorContext),
       {
	 return THStorage_(newWithAllocator)(size, allocator, allocatorContext);
       })
AT_OPS(THStorage*, _new_with_data_and_allocator, Backend::CPU,
       (real* data, long size, THAllocator* allocator, void* allocatorContext),
       {
	 return THStorage_(newWithDataAndAllocator)(data,
						    size,
						    allocator,
						    allocatorContext);
       })
AT_OPS(void, _free, Backend::CPU, (THStorage *storage),
       {
	 THStorage_(free)(storage);
       })
AT_OPS(void, _retain, Backend::CPU, (THStorage *storage),
       {
	 THStorage_(retain)(storage);
       })
AT_OPS(void, _resize, Backend::CPU, (THStorage *storage, int64_t new_size),
       {
	 THStorage_(resize)(storage, new_size);
       })
AT_OPS(void, _fill, Backend::CPU, (THStorage* storage, int value),
       {
	 THStorage_(fill)(storage, value);
       })
AT_OPS(void, _set, Backend::CPU, (THStorage* storage, size_t ind, int value),
       {
	 THStorage_(set)(storage, ind, value);
       })
AT_OPS(int, _get, Backend::CPU, (THStorage* storage, std::size_t ind),
       {
	 return static_cast<int>((THStorage_(get)(storage, ind)));
       })
AT_OPS(void, _set_flag, Backend::CPU, (THStorage* storage, char flag),
       {
	 THStorage_(setFlag)(storage, flag);
       })
AT_OPS(void, _clear_flag, Backend::CPU, (THStorage* storage, char flag),
       {
	 THStorage_(clearFlag)(storage, flag);
       })

} // namespace at

#endif
