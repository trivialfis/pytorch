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
  static void call_deleter(void * ctx, void * data) {
    auto fnptr = (std::function<void(void*)>*) ctx;
    (*fnptr)(data);
    delete fnptr;
  }
  static void* wrapped_alloc(void * ctx, ptrdiff_t size) {
    auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
    ac->retain();
    return ac->allocate(size);
  }
  static void wrapped_free(void * ctx, void * data) {
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
					       nullptr,
  };
  static THCDeviceAllocator wrapped_allocator = {wrapped_alloc,
						 nullptr,
						 wrapped_free,
						 nullptr,
						 nullptr,
  };
  static cudaError_t call_deleter<Backend::CUDA>(void * ctx, void * data) {
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
  static cudaError_t wrapped_free<Backend::CUDA>(void * ctx, void * data) {
    auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
    ac->deallocate(data);
    ac->release();
    return cudaSuccess;
  }  
}

#endif

} // namespace at
