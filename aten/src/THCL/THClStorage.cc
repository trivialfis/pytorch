#include <cstddef>

template <typename T>
struct THClStorage
{
  T *data;
  std::ptrdiff_t size;
  int refcount;
  char flag;
};
