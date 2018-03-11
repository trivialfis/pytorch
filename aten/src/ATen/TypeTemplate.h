#pragma once
#include "ATen/ATenGeneral.h"
#include "ATen/ArrayRef.h"
#include "ATen/Generator.h"
#include "ATen/Half.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/ScalarType.h"
#include "ATen/Scalar.h"
#include "ATen/Allocator.h"
#include "ATen/Type.h"
#include "ATen/Storage.h"

#include "ATen/StorageTemplate.h"

#warning "Remove these headers."
#include "ATen/generated/CPUGenerator.h"
#include "ATen/generated/CPUByteTensor.h"

namespace at{

template <Backend B, ScalarType S>
struct TypeImpl final : public Type
{
  explicit TypeImpl(Context* context);
  virtual ScalarType scalarType() const;
  virtual Backend backend() const override;
  virtual bool is_cuda() const override;
  virtual bool is_sparse() const override;
  virtual bool is_distributed() const override;
  virtual std::unique_ptr<Storage> storage() const override;
  virtual std::unique_ptr<Storage> storage(size_t size) const override;
  virtual std::unique_ptr<Storage>
  storageFromBlob(void * data,
		  int64_t size,
		  const std::function<void(void*)> & deleter=noop_deleter) const override;
  virtual std::unique_ptr<Storage>
  storageWithAllocator(int64_t size,
		       std::unique_ptr<Allocator> allocator) const override;

  virtual std::unique_ptr<Generator> generator() const override;
  virtual const char * toString() const override;
  virtual std::size_t elementSizeInBytes() const override;
  virtual TypeID ID() const override;
  static const char * typeString();
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const override;
  virtual std::unique_ptr<Storage> unsafeStorageFromTH(void * th_pointer, bool retain) const override;
  virtual Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const override;
  Context& get_context() const { return *context; }
};

AT_TEMPLATE TypeImpl<B, S>::TypeImpl(Context *context)
  : Type(context) {}
AT_TEMPLATE ScalarType TypeImpl<B, S>::scalarType() const
{
  return ScalarType::Byte;
}
AT_TEMPLATE Backend TypeImpl<B, S>::backend() const
{
  return B;
}
AT_TEMPLATE bool TypeImpl<B, S>::is_cuda() const
{
  return backend() == kCUDA || backend() == kSparseCUDA;
}
AT_TEMPLATE bool TypeImpl<B, S>::is_sparse() const
{
  return backend() == kSparseCPU || backend() == kSparseCUDA;
}
AT_TEMPLATE bool TypeImpl<B, S>::is_distributed() const
{
  return false;
}
AT_TEMPLATE std::unique_ptr<Storage> TypeImpl<B, S>::storage() const
{
  return std::unique_ptr<Storage>(new StorageType<B, S>(context));
}
AT_TEMPLATE std::unique_ptr<Storage> TypeImpl<B, S>::storage(size_t size) const
{
  return std::unique_ptr<Storage>(new StorageType<B, S>(context, size));
}
AT_TEMPLATE std::unique_ptr<Storage>
TypeImpl<B, S>::storageFromBlob(void *data,
				int64_t size,
				const std::function<void(void*)> & deleter) const
{
  return std::unique_ptr<Storage>(new StorageType<B, S>(context,
						    data,
						    size,
						    deleter));
}
AT_TEMPLATE std::unique_ptr<Storage>
TypeImpl<B, S>::storageWithAllocator(int64_t size,
				     std::unique_ptr<Allocator> allocator) const
{
  return std::unique_ptr<Storage>(new StorageType<B, S>(context, size, std::move(allocator)));
}
AT_TEMPLATE std::unique_ptr<Generator> TypeImpl<B, S>::generator() const
{
#warning "FIXME: Use a not-cpu generator."
  return std::unique_ptr<Generator>(new CPUGenerator{context});
}
AT_TEMPLATE Tensor TypeImpl<B, S>::unsafeTensorFromTH(void * th_pointer,
						      bool retain) const
{
  if (retain)
    ATTHStorage_retain<B, S>::op(static_cast<typename ATTHStorage<B, S>::type*>(th_pointer));
#warning "FIXME: Use not-cpu Tensor."
  return Tensor(new CPUByteTensor(context,(THByteTensor*)(th_pointer)), false);
}
AT_TEMPLATE std::unique_ptr<Storage> TypeImpl<B, S>::unsafeStorageFromTH(void * th_pointer,
									 bool retain) const
{
  if (retain)
    ATTHStorage_retain<B, S>::op(static_cast<typename ATTHStorage<B, S>::type*>(th_pointer));
  return std::unique_ptr<Storage>(new StorageType<B, S>(context,
							static_cast<typename ATTHStorage<B, S>::type*>(th_pointer)));
}
AT_TEMPLATE const char* TypeImpl<B, S>::toString() const
{
  return TypeImpl<B, S>::typeString();
}
AT_TEMPLATE TypeID TypeImpl<B, S>::ID() const
{
#warning "Return a real id"
  return TypeID::CPUByte;
  // return static_cast<int>(B) +
  //   static_cast<int>(S) + static_cast<int>(ScalarType::NumOptions);
}
AT_TEMPLATE std::size_t TypeImpl<B, S>::elementSizeInBytes() const
{
  return sizeof(typename scalar2type<S>::type);
}
AT_TEMPLATE const char *TypeImpl<B, S>::typeString()
{
  return (std::string(backend2string<B>::value) +
	  std::string(scalar2string<S>::value) + "Type").c_str();
}
AT_TEMPLATE Tensor & TypeImpl<B, S>::s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const
{
#warning "FIXME: Unimplmented."
  self.pImpl;
  throw std::runtime_error("Not implemented.");
  return self;
}
} // namespace at
