#include "tensor_types.h"

#include <sstream>
#include <unordered_map>

#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/tensor/python_tensor.h"

using namespace at;

namespace torch { namespace utils {

const char* backend_to_string(const at::Backend backend)
{
  switch (backend)
    {
    case kCPU: return "torch";
    case kCUDA: return "torch.cuda";
    case kCL: return "torch.cl";
    case kSparseCPU: return "torch.sparse";
    case kSparseCUDA: return "torch.cuda.sparse";
    default:
      {
	std::string error =
	  std::string("to_pystr: Unimplemented backend: ") +
	  toStringAll(backend);
	throw std::runtime_error(error);
      }
    }
}

std::string type_to_string(const at::Type& type) {
  std::ostringstream ss;
  ss << backend_to_string(type.backend()) << "." << toString(type.scalarType()) << "Tensor";
  return ss.str();
}

at::Type& type_from_string(const std::string& str) {
  static std::once_flag once;
  static std::unordered_map<std::string, Type*> map;
  std::call_once(once, []() {
    for (auto type : autograd::VariableType::allTypes()) {
      map.emplace(type_to_string(*type), type);
    }
  });

  if (str == "torch.Tensor") {
    return torch::tensor::get_default_tensor_type();
  }

  auto it = map.find(str);
  if (it == map.end()) {
    throw ValueError("invalid type: '%s'", str.c_str());
  }
  return *it->second;
}

std::vector<std::pair<Backend, ScalarType>> all_declared_types() {
  std::vector<std::pair<Backend, ScalarType>> ret;
  std::vector<Backend> backends;
  for (size_t backend = static_cast<size_t>(Backend::Backend_Begin);
       backend != static_cast<size_t>(Backend::Backend_End);
       ++backend)
    {
      Backend backend_enum = static_cast<Backend>(backend);
      if (backend_enum != Backend::Undefined || backend_enum != Backend::NumOptions)
	{
	  backends.push_back(backend_enum);
	}
    }
  // can't easily iterate over enum classes
  std::vector<ScalarType> scalar_types = { ScalarType::Byte, ScalarType::Char, ScalarType::Double, ScalarType::Float,
                                           ScalarType::Int, ScalarType::Long, ScalarType::Short, ScalarType::Half};
  for (auto& backend : backends) {
    for (auto& scalar_type : scalar_types) {
      // there is no sparse half types.
      if (scalar_type == ScalarType::Half && (backend == Backend::SparseCUDA || backend == Backend::SparseCPU)) {
        continue;
      }
      ret.emplace_back(std::make_pair(backend, scalar_type));
    }
  }

  return ret;
}

}} // namespace torch::utils
