#ifndef THCL_GENERAL_INC
#define THCL_GENERAL_INC


#ifdef __cplusplus
# define THCL_EXTERNC extern "C"
#else
# define THCL_EXTERNC extern
#endif

#define THCL_API THCL_EXTERNC

#include <boost/compute/core.hpp>
#include <boost/compute/command_queue.hpp>
// #include "TH/TH.h"
#include "TH/THAllocator.h"
#include "TH/THStorage.h"
#include "TH/THAtomic.h"

namespace compute = boost::compute;

void thcl_wrapped_allocate(void* ptr)
{
  
}

typedef struct _THClDeviceAllocator
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

extern THClState* THCl_global_state();

void THCl_init();
THClState* THClState_alloc();
void THCl_initializeState(THClState *state);

#endif	// THCL_GENERAL_INC
