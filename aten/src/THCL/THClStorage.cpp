#include "THClStorage.h"
#include "THClGeneral.h"
#include "THAtomic.h"
#include "THClKernels.h"

#include "EasyCL.h"
#include "templates/TemplatedKernel.h"
#include "util/StatefulTimer.h"
#include <stdexcept>
#include <iostream>
using namespace std;

static std::string getGetKernelSource();
static std::string getSetKernelSource();

//int state->trace = 0;

// this runs an entire kernel to get one value.  Clearly this is going to be pretty slow, but
// at least it's more or less compatible, and comparable, to how cutorch does it
void THClStorage_set(THClState *state, THClStorage *self, long index, float value)
{
////  cout << "set size=" << self->size << " index=" << index << " value=" << value << endl;
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  THArgCheck(self->wrapper != 0, 1, "storage argument not initialized, is empty");
//  if( self->wrapper->isDeviceDirty() ) { // we have to do this, since we're going to copy it all back again
//                                         // although I suppose we could set via a kernel perhaps
//                                         // either way, this function is pretty inefficient right now :-P
//    if(state->trace) cout << "wrapper->copyToHost() size " << self->size << endl;
//    self->wrapper->copyToHost();
//  }
//  self->data[index] = value;
//  if(state->trace) cout << "wrapper->copyToDevice() size " << self->size << endl;
//  self->wrapper->copyToDevice();

  const char *uniqueName = __FILE__ ":set";
  EasyCL *cl = self->cl; // cant remember if this is a good idea or not :-P
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernel = kernelBuilder.buildKernel( uniqueName, __FILE__, getSetKernelSource(), "THClStorageSet" );
  }

  kernel->inout(self->wrapper);
//  cl_long index2 = (cl_long)index;
//  cout << "index2 " << index2 << endl;
  kernel->in((int)index);
  kernel->in(value);
  kernel->run_1d(1, 1);

  cl->finish(); 
  self->wrapper->markDeviceDirty();

  if(state->addFinish) cl->finish();
}

// this runs an entire kernel to get one value.  Clearly this is going to be pretty slow, but
// at least it's more or less compatible, and comparable, to how cutorch does it
// lgfgs expects a working implementation of this method
float THClStorage_get(THClState *state, const THClStorage *self, long index)
{
////  printf("THClStorage_get\n");
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  THArgCheck(self->wrapper != 0, 1, "storage argument not initialized, is empty");

//  if( self->wrapper->isDeviceDirty() ) {
//    if(state->trace) cout << "wrapper->copyToHost()" << endl;
//    self->wrapper->copyToHost();
//  }
//  return self->data[index];

  const char *uniqueName = __FILE__ ":get";
  EasyCL *cl = self->cl; // cant remember if this is a good idea or not :-P
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernel = kernelBuilder.buildKernel( uniqueName, __FILE__, getGetKernelSource(), "THClStorageGet" );
  }

  float res;
  kernel->out(1, &res);
  kernel->in(self->wrapper);
  kernel->in((int)index);
  kernel->run_1d(1, 1);

  if(state->addFinish) cl->finish();
  return res;
}

THClStorage* THClStorage_new(THClState *state)
{
  return THClStorage_newv2(state, state->currentDevice);
}

THClStorage* THClStorage_newv2(THClState *state, const int device)
{
  THClStorage *storage = (THClStorage*)THAlloc(sizeof(THClStorage));
  storage->device = device;
  storage->cl = THClState_getClv2(state, storage->device);
//  storage->device = -1;
  storage->data = NULL;
  storage->wrapper = 0;
  storage->size = 0;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

THClStorage* THClStorage_newWithSize(THClState *state, int device, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(size > 0)
  {
    StatefulTimer::timeCheck("THClStorage_newWithSize START");
    THClStorage *storage = (THClStorage*)THAlloc(sizeof(THClStorage));
    float *data = new float[size];
    storage->device = device;
    storage->cl = THClState_getClv2(state, storage->device);
    CLWrapper *wrapper = storage->cl->wrap( size, data );
    if(state->trace) cout << "new wrapper, size " << size << endl;
    if(state->trace) cout << "wrapper->createOnDevice()" << endl;
    wrapper->createOnDevice();
    storage->data = data;
    storage->wrapper = wrapper;

    storage->size = size;
    storage->refcount = 1;
    storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
    StatefulTimer::timeCheck("THClStorage_newWithSize END");
    return storage;
  }
  else
  {
    return THClStorage_newv2(state, device);
  }
}

THClStorage* THClStorage_newWithSize1(THClState *state, int device, float data0)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *self = THClStorage_newWithSize(state, 1);
//  THClStorage_set(state, self, 0, data0);
//  return self;
}

THClStorage* THClStorage_newWithSize2(THClState *state, int device, float data0, float data1)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *self = THClStorage_newWithSize(state, 2);
//  THClStorage_set(state, self, 0, data0);
//  THClStorage_set(state, self, 1, data1);
//  return self;
}

THClStorage* THClStorage_newWithSize3(THClState *state, int device, float data0, float data1, float data2)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *self = THClStorage_newWithSize(state, 3);
//  THClStorage_set(state, self, 0, data0);
//  THClStorage_set(state, self, 1, data1);
//  THClStorage_set(state, self, 2, data2);
//  return self;
}

THClStorage* THClStorage_newWithSize4(THClState *state, int device, float data0, float data1, float data2, float data3)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *self = THClStorage_newWithSize(state, 4);
//  THClStorage_set(state, self, 0, data0);
//  THClStorage_set(state, self, 1, data1);
//  THClStorage_set(state, self, 2, data2);
//  THClStorage_set(state, self, 3, data3);
//  return self;
}

THClStorage* THClStorage_newWithMapping(THClState *state, int device, const char *fileName, long size, int isShared)
{
  THError("not available yet for THClStorage");
  return NULL;
}

THClStorage* THClStorage_newWithData(THClState *state, int device, float *data, long size)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *storage = (THClStorage*)THAlloc(sizeof(THClStorage));
//  storage->data = data;
//  storage->size = size;
//  storage->refcount = 1;
//  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
//  return storage;
}

void THClStorage_retain(THClState *state, THClStorage *self)
{
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&self->refcount);
}

void THClStorage_free(THClState *state, THClStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (THAtomicDecrementRef(&self->refcount))
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      StatefulTimer::timeCheck("THClStorage_free START");
      if(state->trace && self->size > 0) cout << "delete wrapper, size " << self->size << endl;
      delete self->wrapper;
      delete self->data;
      StatefulTimer::timeCheck("THClStorage_newWithSize END");
    }
    THFree(self);
  }
}
void THClStorage_fill(THClState *state, THClStorage *self, float value)
{
 StatefulTimer::timeCheck("THClStorage_fill START");
  for( int i = 0; i < self->size; i++ ) {
    self->data[i] = value;
  }
  self->wrapper->copyToDevice();
  if(state->trace) cout << "wrapper->copyToDevice() size" << self->size << endl;
  StatefulTimer::timeCheck("THClStorage_fill END");
}

void THClStorage_resize(THClState *state, THClStorage *self, long size)
{
  StatefulTimer::timeCheck("THClStorage_resize START");
  if( size <= self->size ) {
    return;
  }
  delete self->wrapper;
  if(state->trace && self->size > 0) cout << "delete wrapper" << endl;
  delete[] self->data;
  self->data = new float[size];
  EasyCL *cl = self->cl;
  self->wrapper = cl->wrap( size, self->data );
  self->wrapper->createOnDevice();
    if(state->trace) cout << "new wrapper, size " << size << endl;
  self->size = size;
  StatefulTimer::timeCheck("THClStorage_resize END");
}

std::string getGetKernelSource() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClStorageGet.cl" )
  // ]]]
  // generated using cog, from THClStorageGet.cl:
  const char * kernelSource =  
  "kernel void THClStorageGet(global float *res, global float *data, int index) {\n" 
  "  if(get_global_id(0) == 0) {\n" 
  "    res[0] = data[index];\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

std::string getSetKernelSource() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClStorageSet.cl" )
  // ]]]
  // generated using cog, from THClStorageSet.cl:
  const char * kernelSource =  
  "kernel void THClStorageSet(global float *data, int index, float value) {\n" 
  "  if(get_global_id(0) == 0) {\n" 
  "//    int index2 = index;\n" 
  "//    data[index2] = 44;\n" 
  "    data[index] = value;\n" 
  "//    data[2] = index2;\n" 
  "//    data[3] = value;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

