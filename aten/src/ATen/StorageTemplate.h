#pragma once

#include <TH/TH.h>
#include <THNN/THNN.h>
#undef THNN_
#include <THS/THS.h>

#include "ATen/Storage.h"
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

template <Backend B, ScalarType S>
struct StorageDispatch;
#define AT_CREATE_CPU_STORAGE_DISPATCH(_1, SCALAR, _2)		\
  template <>							\
  struct StorageDispatch<Backend::CPU, ScalarType::SCALAR>	\
  {								\
    using StorageType = TH ## SCALAR ## Storage;		\
  };
#define AT_CREATE_CUDA_STORAGE_DISPATCH(_1, SCALAR, _2)		\
  template <>							\
  struct StorageDispatch<Backend::CUDA, ScalarType::SCALAR>	\
  {								\
    using StorageType = THCuda ## SCALAR ## Storage;		\
  };
#define AT_CREATE_CL_STORAGE_DISPATCH(_1, SCALAR, _2)	        \
  template <>							\
  struct StorageDispatch<Backend::CL, ScalarType::SCALAR>	\
  {								\
    using StorageType = THCL ## SCALAR ## Storage;		\
  };

AT_FORALL_SCALAR_TYPES(AT_CREATE_CPU_STORAGE_DISPATCH)
#if AT_CUDA_ENABLED()
AT_FORALL_SCALAR_TYPES(AT_CREATE_CUDA_STORAGE_DISPATCH)
#endif
#if AT_CL_ENABLED()
AT_FORALL_SCALAR_TYPES(AT_CREATE_CL_STORAGE_DISPATCH)
#endif


template <Backend B, ScalarType S>
struct _Storage final : public Storage
{
public:
  explicit _Storage(Context* context);

  _Storage(Context* context,
	   typename StorageDispatch<B, S>::StorageType *wrapped);
  _Storage(Context* context, std::size_t size);
  _Storage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  _Storage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~_Storage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual _Storage& retain() override;
  virtual _Storage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual _Storage& resize(int64_t new_size) override;
  virtual _Storage& fill(Scalar value) override;
  virtual _Storage& set(std::size_t ind, Scalar value) override;
  virtual _Storage& fast_set(std::size_t ind, Scalar value) override;
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
  ${THStorage} *storage;
  Context* context;
};
}
