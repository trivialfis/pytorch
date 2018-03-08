#include "THClGeneral.h"

void THCl_init(THClState *state)
{
  state->addFinish = 0;
  state->allocatedDevices = 0;
}

THClState* THClState_alloc()
{
  THClState* state = (THClState*) malloc(sizeof(THClState));
  memset(state, 0, sizeof(THClState));
  state->initialized = 1;
  return state;
}
