#include "THClAllocator.h"

static void* THClHostAllocator_malloc(void *ctx, ptrdiff_t size)
{
  void *ptr;
  if (size < 0) THError("Invalid memory size %d", size);
  if (size == 0) return NULL;
  ptr = (void*)malloc(size);
  if (!ptr)
    {
      THError("Memory allocation failed");
    }
  return ptr;
}

static void THClHostAllocator_free(void* ctx, void* ptr)
{
  if (!ptr) return;
  free(ptr);
}

static void* THClHostAllocator_realloc(void *ctx, void *ptr, ptrdiff_t size)
{
  if (size < 0) THError("Invalid memory size %d", size);
  if (size == 0)
    {
      THClHostAllocator_free(NULL, ptr);
      return NULL;
    }
  ptr = realloc(ptr, size);
  return ptr;
}

THAllocator THClHostAllocator =
  {
   &THClHostAllocator_malloc,
   &THClHostAllocator_realloc,
   &THClHostAllocator_free
  };
