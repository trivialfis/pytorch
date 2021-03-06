#pragma once

#include <Python.h>
#include <ATen/ATen.h>
#include <utility>
#include <vector>

namespace torch { namespace utils {

std::string type_to_string(const at::Type& type);
at::Type& type_from_string(const std::string& str);
const char* backend_to_string(const at::Backend backend);

// return a vector of all "declared" types, even those that weren't compiled
std::vector<std::pair<at::Backend, at::ScalarType>> all_declared_types();

}} // namespace torch::utils
