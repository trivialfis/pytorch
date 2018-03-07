#ifndef THCL_GENERAL_INC
#define THCL_GENERAL_INC

#include <boost/compute/core.hpp>

#ifdef __cplusplus
# define THC_EXTERNC extern "C"
#else
# define THC_EXTERNC extern
#endif

#define THCL_API THCL_EXTERNC

namespace compute = boost::compute;

typedef struct THCLState
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

 // EasyCL *getCl();  
} THCLState;

void THCL_init(THCLState *state);
THCLState* THCLState_alloc(void);
void THCL_initializeState(THCLState *state);

#endif	// THCL_GENERAL_INC
