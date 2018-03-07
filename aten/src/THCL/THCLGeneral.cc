#include "THCLGeneral.h"

void THCL_init(THCLState *state)
{
  state->addFinish = 0;
  state->allocatedDevices = 0;
}

THCLState* THCLState_alloc(void)
{
  THCLState* state = (THCLState*) malloc(sizeof(THCLState));
  memset(state, 0, sizeof(THCLState));
  return state;
}

void THCL_initializeState(THCLState *state)
{
  state->initialized = 1;
  state->device = compute::system::default_device();
}
