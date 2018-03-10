#ifndef THCL_GENERAL_INC
#define THCL_GENERAL_INC

#include <boost/compute/core.hpp>

#ifdef __cplusplus
# define THCL_EXTERNC extern "C"
#else
# define THCL_EXTERNC extern
#endif

#define THCL_API THCL_EXTERNC
#include <boost/compute/command_queue.hpp>
#include "TH/TH.h"

namespace compute = boost::compute;

typedef struct THClDeviceAllocator
{
  cl_int (*malloc)(void*, void**, size_t, cl_command_queue);
  cl_int (*realloc)(void*, void**, size_t, size_t, cl_command_queue);
  cl_int (*free)(void*, void*);
} THClDeviceAllocator;

typedef struct THClState
{
  int initialized;

  compute::device* currentDevice;
  compute::command_queue* currentQueue;
  compute::context* currentContext;
  size_t numDevices;

  THAllocator*  clHostAllocator;
  THAllocator* clDeviceAllocator;

} THClState;

void THCl_init(THClState *state);
THClState* THClState_alloc();
void THCl_initializeState(THClState *state);

#endif	// THCL_GENERAL_INC
