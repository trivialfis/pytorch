#include "THClGeneral.h"
#include "THClAllocator.h"
#include <boost/compute/context.hpp>
#include <clBLAS.h>

// static cl_int clMallocWrapper(void *ctx, void **devPtr, size_t size, cl_command_queue queue)
// {

// }
THClState* THCl_global_state()
{
  static THClState *state = THClState_alloc();
  return state;
}

void THCl_init()
{
  THClState *state = THCl_global_state();
  if (!state->clHostAllocator)
    {
      state->clHostAllocator = &THClHostAllocator;
    }

  int numDevices = 0;
  numDevices = compute::system::device_count();
  state->numDevices = numDevices;

  compute::device default_device = compute::system::default_device();
  compute::device* device = new compute::device(default_device);
  state->currentDevice = device;

  compute::context* context = new compute::context{*device};
  state->currentContext = context;
  compute::command_queue* queue = new compute::command_queue(*context, *device);
  state->currentQueue = new compute::command_queue();

  cl_int errno;
  errno = clblasSetup();
  if(errno != CL_SUCCESS)
    {
      THError("clblas initialize failed.");
    }
}

THClState* THClState_alloc()
{
  THClState* state = (THClState*) malloc(sizeof(THClState));
  memset(state, 0, sizeof(THClState));
  state->initialized = 1;
  return state;
}
