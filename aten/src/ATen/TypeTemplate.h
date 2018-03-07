#include "ATen/ATenGeneral.h"
#include "ATen/ArrayRef.h"
#include "ATen/Generator.h"
#include "ATen/Half.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/ScalarType.h"
#include "ATen/Scalar.h"
// #include "ATen/Tensor.h"
#include "ATen/Allocator.h"
#include "ATen/Type.h"

namespace at{
template <Backend b, ScalarType s>
struct ID_op
{
  TypeID ID() {throw std::runtime_error("Not implemented");}
};
// Put in here for latter replacing those duplicated.
template <Backend b, ScalarType s>
struct TypeD
{
  int ID() {return static_cast<int>(b) + static_cast<int>(s) + static_cast<int>(ScalarType::NumOptions);}
};
template <>
struct ID_op<Backend::CL, ScalarType::Byte>
{
  TypeID ID() {return TypeID::CLByte;}
};
template <>
struct ID_op<Backend::CL, ScalarType::Char>
{
  TypeID ID() {return TypeID::CLChar;}
};
template <>
struct ID_op<Backend::CL, ScalarType::Double>
{
  TypeID ID() {return TypeID::CLDouble;}
};
template <>
struct ID_op<Backend::CL, ScalarType::Float>
{
  TypeID ID() {return TypeID::CLFloat;}
};
template <>
struct ID_op<Backend::CL, ScalarType::Half>
{
  TypeID ID() {return TypeID::CLHalf;}
};
template <>
struct ID_op<Backend::CL, ScalarType::Int>
{
  TypeID ID() {return TypeID::CLInt;}
};
template <>
struct ID_op<Backend::CL, ScalarType::Long>
{
  TypeID ID() {return TypeID::CLLong;}
};
template <>
struct ID_op<Backend::CL, ScalarType::Short>
{
  TypeID ID() {return TypeID::CLShort;}
};

#define DEFINE_TYPE_METHOD(RETURN, NAME, ...)		\
  template <Backend B, ScalarType S>			\
  struct NAME ## _op					\
  {							\
    RETURN NAME	()					\
      __VA_ARGS__					\
  };

DEFINE_TYPE_METHOD(ScalarType,
		   scalarType,
		   {
		     return S;
		   })
DEFINE_TYPE_METHOD(bool, is_cuda,
		   {
		     return B == kCUDA || B == kSparseCUDA;
		   })
DEFINE_TYPE_METHOD(bool, is_sparse,
		   {
		     return B == kSparseCPU || B == kSparseCUDA;
		   })
DEFINE_TYPE_METHOD(bool, is_distributed,
		   {
		     return false;
		   })

template <Backend B>
struct dispatch_storage;
template <>
struct dispatch_storage<Backend::CPU>
{
  
}

template <Backend B, ScalarType S>
struct _Type final : public Type, ID_op<B, S>, scalarType_op<B, S>
{
  explicit _Type(Context* context)
    : Type(context) {}

  virtual ~_Type() {}
  static void registerAll(Context * context);
  virtual std::unique_ptr<Storage> storage() const override;
  virtual std::unique_ptr<Storage> storage(size_t size) const override;
  virtual std::unique_ptr<Storage> storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter=noop_deleter) const override;
  virtual std::unique_ptr<Storage> storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const override;
  virtual std::unique_ptr<Generator> generator() const override;
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const override;
  virtual std::unique_ptr<Storage> unsafeStorageFromTH(void * th_pointer, bool retain) const override;
  virtual const char * toString() const override;
  virtual std::size_t elementSizeInBytes() const override;
  virtual Type & toBackend(Backend b) const;
  virtual Type & toScalarType(ScalarType s) const;
  Context& get_context() const { return *context; }
};

using CLByteType = _Type<Backend::CL, ScalarType::Byte>;
using CLCharType = _Type<Backend::CL, ScalarType::Char>;
using CLDoubleType = _Type<Backend::CL, ScalarType::Double>;
using CLFloatType = _Type<Backend::CL, ScalarType::Float>;
using CLIntType = _Type<Backend::CL, ScalarType::Int>;
using CLLongType = _Type<Backend::CL, ScalarType::Long>;
using CLShortType = _Type<Backend::CL, ScalarType::Short>;

} // namespace at
