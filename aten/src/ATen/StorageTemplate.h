#pragma once

#include <TH/TH.h>
#include <THNN/THNN.h>
#undef THNN_
#include <THS/THS.h>

#include "ATen/StorageBase.h"
#include "ATen/Context.h"

#include <memory>
#include <string>

#include <ATen/Config.h>

namespace at
{
struct Allocator;

// FIXME: string mapping should be put into ScalarType.h and Type.h
template <ScalarType s>
struct scalar2string;
#define AT_SCALAR_MAPPING(_1, n, _2)		\
  template <>					\
  struct scalar2string<ScalarType::n>		\
  {						\
    char constexpr static value[] = #n;		\
  };
AT_FORALL_SCALAR_TYPES(AT_SCALAR_MAPPING)

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


AT_TEMPLATE struct Storage final : public StorageBase
{
public:
  explicit Storage(Context* context);

  Storage(Context* context,
	   typename StorageDispatch<B, S>::StorageType *wrapped);
  Storage(Context* context, std::size_t size);
  Storage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  Storage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~Storage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual Storage& retain() override;
  virtual Storage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual Storage& resize(int64_t new_size) override;
  virtual Storage& fill(Scalar value) override;
  virtual Storage& set(std::size_t ind, Scalar value) override;
  virtual Storage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;
  {
    return std::string(backend2string<B>)
      + std::string(scalar2string<S>) + "Storage";
  }
  static const char * typeString() const
  {
    return std::string(backend2string<B>)
      + std::string(scalar2string<S>) + "Type";
  }

protected:
  friend struct ${Type};
  ATStorage *storage;
  Context* context;
};

AT_TEMPLATE Storage::Storage(Context* context)
  :storage(ATTHStorage_new<B, S>(), context){}
AT_TEMPLATE Storage::Storage(Context* context, AT_STORAGE_TYPE* storage)
  :storage(storage), context(context) {}
AT_TEMPLATE Storage::Storage(Context* context, std::size_t storage_size)
  :storage(storage), context(context) {}

template <Backend b>
struct ATTHAllocator;

template <>
struct ATTHAllocator<Backend::CPU>
{
  static THAllocator storage_deleter = {nullptr,
					nullptr,
					call_deleter,};
  static THAllocator wrapped_allocator = {wrapped_alloc,
					  nullptr,
					  wrapped_free,};
  static void call_deleter(void * ctx, void * data)
  {
    auto fnptr = (std::function<void(void*)>*) ctx;
    (*fnptr)(data);
    delete fnptr;
  }
  static void* wrapped_alloc(void * ctx, ptrdiff_t size)
  {
    auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
    ac->retain();
    return ac->allocate(size);
  }
  static void wrapped_free(void * ctx, void * data)
  {
    auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
    ac->deallocate(data);
    ac->release();
  }
};

# if AT_CUDA_ENABLED()

template <>
struct ATTHAllocator<Backend::CUDA>
{
  static THCDeviceAllocator storage_deleter = {nullptr,
					       nullptr,
					       call_deleter,
					       nullptr,
					       nullptr,};
  static THCDeviceAllocator wrapped_allocator = {wrapped_alloc,
						 nullptr,
						 wrapped_free,
						 nullptr,
						 nullptr,};
  static cudaError_t call_deleter<Backend::CUDA>(void * ctx, void * data)
  {
    auto fnptr = (std::function<void(void*)>*) ctx;
    (*fnptr)(data);
    delete fnptr;
    return cudaSuccess;
  }
  static cudaError_t wrapped_alloc<Backend::CUDA>(void * ctx,
						  void** result,
						  size_t size,
						  cudaStream_t stream)
  {
    auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
    ac->retain();
    *result = ac->allocate(size);
    return cudaSuccess;
  }
  static cudaError_t wrapped_free<Backend::CUDA>(void * ctx, void * data)
  {
    auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
    ac->deallocate(data);
    ac->release();
    return cudaSuccess;
  }  
}

#endif

Storage::Storage(Context* context, std::size_t size,
		 std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context)
{
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = ATTHStorage_newWithAllocator(size,
					 &ATTHAllocator<B>::wrapped_allocator,
					 ctx);
  ctx->release();
  ATTHStorage_clearFlag(storage, TH_STORAGE_RESIZABLE);
}

Storage::Storage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(ATTHStorage_newWithDataAndAllocator<B, S>
	    (static_cast<uint8_t*>(data), size,
	     &storage_deleter,
	     new std::function<void(void*)>(deleter))),
    context(context)
{
  ATTHStorage_clearFlag<B, S>( storage, TH_STORAGE_RESIZABLE);
}

Storage::~Storage() {
  ATTHStorage_free<B, S>( storage);
}

std::size_t Storage::elementSize() const {
  return sizeof(uint8_t);
}

std::size_t Storage::size() const {
  return storage->size;
}

void* Storage::data() {
  return storage->data;
}

const void* Storage::data() const {
  return storage->data;
}

auto Storage::retain() -> Storage&
  {
   ATTHStorage_retain<B, S>( storage);
   return *this;
  }

auto Storage::free() -> Storage&
  {
   ATTHStorage_free<B, S>( storage);
   return *this;
  }

void* Storage::unsafeGetTH(bool retain) const
{
  if (retain) {
    ATTHStorage_retain(storage);
  }
  return storage;
}

auto Storage::resize(int64_t new_size) -> Storage&
  {
   ATTHStorage_resize(storage, new_size);
   return *this;
  }

auto Storage::fill(Scalar value) -> Storage&
  {
   ATTHStorage_fill<B, S>(storage, (value.toByte()));
   return *this;
  }

auto Storage::set(std::size_t ind, Scalar value) -> Storage&
  {
   ATTHStorage_set<B, S>(storage, ind, (value.toByte()));
   return *this;
  }

auto Storage::fast_set(std::size_t ind, Scalar value) -> Storage&
  {
   throw std::runtime_error("unsupported operation 'fast_set'");
  }

auto Storage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<uint8_t>((ATTHStorage_get<B, S>( storage, ind)));
}

auto Storage::fast_get(std::size_t ind) -> Scalar {
  if(B == Backend::CPU)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<uint8_t>((storage->data[ind]));
}

void Storage::set_flag(char flag) {
  ATTHStorage_setFlag<B, S>( storage, flag);
}

void Storage::clear_flag(char flag) {
  ATTHStorage_clearFlag<B, S>(storage, flag);
}

int Storage::getDevice() const {
  if (B == Backend::CPU)
    throw std::runtime_error("CPU storage has no device"); //storage->device;
  else
    return storage->device;
}

Type& Storage::type() const {
  return context->getType(B, S);
}

} // namespace at
