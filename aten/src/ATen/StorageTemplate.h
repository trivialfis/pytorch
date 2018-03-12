#pragma once

#include <TH/TH.h>
// #include <THNN/THNN.h>
// #undef THNN_
// #include "THCL/THCl.h"
#include <THS/THS.h>

#include "ATen/StorageOps.h"
#include "ATen/Storage.h"
#include "ATen/Context.h"

#include <memory>
#include <string>

#include <ATen/Config.h>
#include <ATen/ATenGeneral.h>

namespace at
{
struct Allocator;
template <Backend B>
static void call_deleter(void * ctx, void * data);
template <Backend B>
void* wrapped_alloc(void * ctx, ptrdiff_t size);
template <Backend B>
static void wrapped_free(void * ctx, void * data);

// FIXME: string mapping should be put into ScalarType.h and Type.h
// FIXME: Use template technique instead of macro.
template <ScalarType s>
struct scalar2string;
#define AT_CREATE_SCALAR2STRING(_1, n, _2)	\
  template <>					\
  struct scalar2string<ScalarType::n>		\
  {						\
    char constexpr static value[] = #n;		\
  };
AT_FORALL_SCALAR_TYPES(AT_CREATE_SCALAR2STRING)

template <ScalarType s>
struct scalar2type;
#define AT_CREATE_SCALAR2TYPE(t, n, _2)		\
  template <>					\
  struct scalar2type<ScalarType::n>		\
  {						\
    using type = t;				\
  };
AT_FORALL_SCALAR_TYPES(AT_CREATE_SCALAR2TYPE)

template <Backend B>
struct backend2string;
#define AT_BACKEND_MAPPING(b)			\
  template <>					\
  struct backend2string<Backend::b>		\
  {						\
    char constexpr static value[] = #b;		\
  };
AT_BACKEND_MAPPING(CPU)
AT_BACKEND_MAPPING(SparseCPU)
AT_BACKEND_MAPPING(CL)
AT_BACKEND_MAPPING(CUDA)
AT_BACKEND_MAPPING(SparseCUDA)

AT_TEMPLATE struct StorageType final : public Storage
{
public:
  explicit StorageType(Context* context);

  StorageType(Context* context, AT_STORAGE_TYPE* wrapped);
  StorageType(Context* context, std::size_t size);
  StorageType(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  StorageType(Context* context,
	      void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~StorageType();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual StorageType& retain() override;
  virtual StorageType& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual StorageType& resize(int64_t new_size) override;
  virtual StorageType& fill(Scalar value) override;
  virtual StorageType& set(std::size_t ind, Scalar value) override;
  virtual StorageType& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override
  {
    return (std::string(backend2string<B>::value)
	    + std::string(scalar2string<S>::value)
	    + std::string("Storage")).c_str();
  }
  static const char * typeString()
  {
    return (std::string(backend2string<B>::value)
	    + std::string(scalar2string<S>::value)
	    + std::string("Type")).c_str();
  }

protected:
  // friend struct ${Type};
  AT_STORAGE_TYPE *storage;
  Context* context;
};

AT_TEMPLATE StorageType<B, S>::StorageType(Context* context)
  :storage{ATTHStorage_new<B, S>::op()}, context(context){}
AT_TEMPLATE StorageType<B, S>::StorageType(Context* context, AT_STORAGE_TYPE* storage)
  :storage(storage), context(context) {}
AT_TEMPLATE StorageType<B, S>::StorageType(Context* context, std::size_t storage_size)
  :storage(storage), context(context) {}

template <Backend b>
struct ATAllocator;

template <>
void call_deleter<Backend::CPU>(void * ctx, void * data)
{
  auto fnptr = (std::function<void(void*)>*) ctx;
  (*fnptr)(data);
  delete fnptr;
}
template <>
void* wrapped_alloc<Backend::CPU>(void * ctx, ptrdiff_t size)
{
  auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
  ac->retain();
  return ac->allocate(size);
}
template <>
void wrapped_free<Backend::CPU>(void * ctx, void * data)
{
  auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
  ac->deallocate(data);
  ac->release();
}

template <>
struct ATAllocator<Backend::CPU>
{
  THAllocator storage_deleter;
  THAllocator wrapped_allocator;
  ATAllocator() :
    storage_deleter {nullptr,
		     nullptr,
		     &call_deleter<Backend::CPU>,},
    wrapped_allocator {&wrapped_alloc<Backend::CPU>,
		       nullptr,
		       &wrapped_free<Backend::CPU>,} {}
  ~ATAllocator() {}
};

# if AT_CUDA_ENABLED()
template <>
cudaError_t call_deleter<Backend::CUDA>(void * ctx, void * data)
{
  auto fnptr = (std::function<void(void*)>*) ctx;
  (*fnptr)(data);
  delete fnptr;
  return cudaSuccess;
}
template <>
cudaError_t wrapped_alloc<Backend::CUDA>(void * ctx,
					 void** result,
					 size_t size,
					 cudaStream_t stream)
{
  auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
  ac->retain();
  *result = ac->allocate(size);
  return cudaSuccess;
}
template <>
cudaError_t wrapped_free<Backend::CUDA>(void * ctx, void * data)
{
  auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
  ac->deallocate(data);
  ac->release();
  return cudaSuccess;
}
template <>
struct ATAllocator<Backend::CUDA>
{
  THCDeviceAllocator storage_deleter ;
  THCDeviceAllocator wrapped_allocator;
  ATAllocator() :
    storage_deleter {nullptr,
		     nullptr,
		     call_deleter<Backend::CUDA>,
		     nullptr,
		     nullptr},
    wrapped_allocator {wrapped_alloc<Backend::CUDA>,
		       nullptr,
		       wrapped_free<Backend::CUDA>,
		       nullptr,
		       nullptr}{}
   ~ATAllocator() {}
};

#endif
#if AT_CL_ENABLED()
template <>
struct ATAllocator<Backend::CL>
{
  THClDeviceAllocator storage_deleter;
  THClDeviceAllocator wrapped_allocator;
  ATAllocator() :
    storage_deleter {nullptr,
		     nullptr,
		     nullptr},
    wrapped_allocator {nullptr,
		       nullptr,
		       nullptr} {}
  ~ATAllocator() {}
// #error "Not implemented."
};
#endif

template <Backend B>
ATAllocator<B>* _global_allocator()
{
  static ATAllocator<B> allocator;
  return &allocator;
}


AT_TEMPLATE StorageType<B, S>::StorageType(Context* context, std::size_t size,
					   std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context)
{
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = ATTHStorage_new_with_allocator
    <B, S>::op(size,
	       &_global_allocator<B>()->wrapped_allocator,
	       ctx);
  ctx->release();
  ATTHStorage_clear_flag<B, S>::op(storage, TH_STORAGE_RESIZABLE);
}
AT_TEMPLATE
StorageType<B, S>::StorageType(Context* context,
			       void * data, std::size_t size,
			       const std::function<void(void*)> & deleter)
  : storage(ATTHStorage_new_with_data_and_allocator<B, S>::op
	    (static_cast<typename scalar2type<S>::type*>(data), size,
	     &_global_allocator<B>()->storage_deleter,
	     new std::function<void(void*)>(deleter))),
    context(context)
{
  ATTHStorage_clear_flag<B, S>::op(storage, TH_STORAGE_RESIZABLE);
}
AT_TEMPLATE
StorageType<B, S>::~StorageType() {
  ATTHStorage_free<B, S>::op(storage);
}
AT_TEMPLATE std::size_t StorageType<B, S>::elementSize() const
{
  return sizeof(uint8_t);
}
AT_TEMPLATE std::size_t StorageType<B, S>::size() const
{
  return storage->size;
}
AT_TEMPLATE void* StorageType<B, S>::data() {
  return storage->data;
}
AT_TEMPLATE const void* StorageType<B, S>::data() const
{
  return storage->data;
}
AT_TEMPLATE auto StorageType<B, S>::retain() -> StorageType&
  {
   ATTHStorage_retain<B, S>::op(storage);
   return *this;
  }
AT_TEMPLATE auto StorageType<B, S>::free() -> StorageType&
  {
   ATTHStorage_free<B, S>::op(storage);
   return *this;
  }
AT_TEMPLATE void* StorageType<B, S>::unsafeGetTH(bool retain) const
{
  if (retain) {
    ATTHStorage_retain<B, S>::op(storage);
  }
  return storage;
}
AT_TEMPLATE auto StorageType<B, S>::resize(int64_t new_size) -> StorageType&
  {
   ATTHStorage_resize<B, S>::op(storage, new_size);
   return *this;
  }
AT_TEMPLATE auto StorageType<B, S>::fill(Scalar value) -> StorageType&
  {
   ATTHStorage_fill<B, S>::op(storage, (value.toByte()));
   return *this;
  }
AT_TEMPLATE auto StorageType<B, S>::set(std::size_t ind, Scalar value) -> StorageType&
  {
   ATTHStorage_set<B, S>::op(storage, ind, (value.toByte()));
   return *this;
  }
AT_TEMPLATE auto StorageType<B, S>::fast_set(std::size_t ind,
					     Scalar value) -> StorageType&
  {
   throw std::runtime_error("unsupported operation 'fast_set'");
  }
AT_TEMPLATE auto StorageType<B, S>::get(std::size_t ind) -> Scalar
{
  // static cast to fix  long -> int64_t issues
  return static_cast<typename scalar2type<S>::type>((ATTHStorage_get<B, S>::op(storage, ind)));
}
AT_TEMPLATE auto StorageType<B, S>::fast_get(std::size_t ind) -> Scalar
{
  if(B == Backend::CPU)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<typename scalar2type<S>::type>((storage->data[ind]));
}
AT_TEMPLATE void StorageType<B, S>::set_flag(char flag)
{
  ATTHStorage_set_flag<B, S>::op(storage, flag);
}
AT_TEMPLATE void StorageType<B, S>::clear_flag(char flag)
{
  ATTHStorage_clear_flag<B, S>::op(storage, flag);
}
AT_TEMPLATE int StorageType<B, S>::getDevice() const
{
  if (B == Backend::CPU)
    throw std::runtime_error("CPU storage has no device");
  else
    return storage->device;
}
AT_TEMPLATE Type& StorageType<B, S>::type() const
{
  return context->getType(B, S);
}
// FIXME: Define here to fool other class for underlying changes.
// Remove them when code generation with python is out of site and mind.
#define AT_CPU_TEMP_STORAGE_TYPE(_1, name, _2)				\
  using CPU ## name ## Storage = StorageType<Backend::CPU, ScalarType::name>;

// AT_FORALL_SCALAR_TYPES(AT_CPU_TEMP_STORAGE_TYPE)

} // namespace at
