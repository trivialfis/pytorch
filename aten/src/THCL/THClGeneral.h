#ifndef THCL_GENERAL_INC
#define THCL_GENERAL_INC

#include <boost/compute/core.hpp>

#ifdef __cplusplus
# define THCL_EXTERNC extern "C"
#else
# define THCL_EXTERNC extern
#endif

#define THCL_API THCL_EXTERNC

namespace compute = boost::compute;

class THClDeviceAllocator
{
  compute::buffer malloc()
  {
    return compute::buffer();
  }
};

typedef struct THClState
{
  int initialized;
  int allocatedDevices;
  int currentDevice;
  int trace; // default 0; set to 1 to see message for every gpu buffer alloc, delete,
             // or device <-> host transfer
  int addFinish; // default 0, should we add clFinish() after any kernel, enqueue, etc?
                 // (good for debugging stuff, bad for perf)
  int detailedTimings;
  compute::device device;
  THClDeviceAllocator* clDeviceAllocator;

} THClState;

void THCl_init(THClState *state);
THClState* THClState_alloc();
void THCl_initializeState(THClState *state);

#endif	// THCL_GENERAL_INC
