#include <iostream>

#include "THClStorageCopy.h"
#include "THClGeneral.h"

#include <stdio.h>
#include "EasyCL.h"

using namespace std;

void THClStorage_rawCopy(THClState *state, THClStorage *self, float *src)
{
  THError("not available yet for THClStorage");
//  THClCheck(clMemcpyAsync(self->data, src, self->size * sizeof(float), clMemcpyDeviceToDevice, THClState_getCurrentStream(state)));
}

void THClStorage_copy(THClState *state, THClStorage *self, THClStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  if( !self->wrapper->isOnDevice() ) {
    self->wrapper->createOnDevice();
  }
  src->wrapper->copyTo( self->wrapper );
  if(state->trace) cout << "wrapper->copyTo() size" << self->size << endl;
}

void THClStorage_copyCl(THClState *state, THClStorage *self, THClStorage *src)
{
  THError("not available yet for THClStorage");
  THArgCheck(self->size == src->size, 2, "size does not match");
//  THClCheck(clMemcpyAsync(self->data, src->data, self->size * sizeof(float), clMemcpyDeviceToDevice, THClState_getCurrentStream(state)));
}

void THClStorage_copyFloat(THClState *state, THClStorage *self, struct THFloatStorage *src)
{
//  cout << "THClStorgae_copyFloat()" << endl;
  THArgCheck(self->size == src->size, 2, "size does not match");
  for( int i = 0; i < self->size; i++ ) {
    self->data[i] = src->data[i];
  }
  self->wrapper->copyToDevice();
  if(state->trace) cout << "wrapper->copyToDevice() size" << self->size << endl;
 // THClCheck(clMemcpy(self->data, src->data, self->size * sizeof(float), clMemcpyHostToDevice));
}

#define TH_CL_STORAGE_IMPLEMENT_COPY(TYPEC)                           \
  void THClStorage_copy##TYPEC(THClState *state, THClStorage *self, struct TH##TYPEC##Storage *src) \
  {                                                                     \
    THFloatStorage *buffer;                                             \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    buffer = THFloatStorage_newWithSize(src->size);                     \
    THFloatStorage_copy##TYPEC(buffer, src);                            \
    THClStorage_copyFloat(state, self, buffer);                              \
    THFloatStorage_free(buffer);                                        \
  }

TH_CL_STORAGE_IMPLEMENT_COPY(Byte)
TH_CL_STORAGE_IMPLEMENT_COPY(Char)
TH_CL_STORAGE_IMPLEMENT_COPY(Short)
TH_CL_STORAGE_IMPLEMENT_COPY(Int)
TH_CL_STORAGE_IMPLEMENT_COPY(Long)
TH_CL_STORAGE_IMPLEMENT_COPY(Double)

void THFloatStorage_copyCl(THClState *state, THFloatStorage *self, struct THClStorage *src)
{
//  cout << "THfloatStorage_copyCl" << endl;
  THArgCheck(self->size == src->size, 2, "size does not match");
  if( src->size == 0 ) {
    // dont need to do anything...
    return;
  }
  if( src->wrapper->isDeviceDirty() ) {
    src->wrapper->copyToHost();
    if(state->trace) cout << "wrapper->copyToHost() size" << self->size << endl;
  }
  for( int i = 0; i < self->size; i++ ) {
    self->data[i] = src->data[i];
  }
}

#define TH_CL_STORAGE_IMPLEMENT_COPYTO(TYPEC)                           \
  void TH##TYPEC##Storage_copyCl(THClState *state, TH##TYPEC##Storage *self, struct THClStorage *src) \
  {                                                                     \
    THFloatStorage *buffer;                                             \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    buffer = THFloatStorage_newWithSize(src->size);                     \
    THFloatStorage_copyCl(state, buffer, src);                               \
    TH##TYPEC##Storage_copyFloat(self, buffer);                         \
    THFloatStorage_free(buffer);                                        \
  }

TH_CL_STORAGE_IMPLEMENT_COPYTO(Byte)
TH_CL_STORAGE_IMPLEMENT_COPYTO(Char)
TH_CL_STORAGE_IMPLEMENT_COPYTO(Short)
TH_CL_STORAGE_IMPLEMENT_COPYTO(Int)
TH_CL_STORAGE_IMPLEMENT_COPYTO(Long)
TH_CL_STORAGE_IMPLEMENT_COPYTO(Double)

