#ifndef THCL_GENERIC_FILE
#define THCL_GENERIC_FILE "generic/THClStorage.cc"
#else

#include <boost/compute/core.hpp>
#include <boost/compute/command_queue.hpp>

real* THClStorage_(data)(THClState *state, const THClStorage *self)
{
  return self->data;
}

ptrdiff_t THClStorage_(size)(THClState *state, const THClStorage *self)
{
  return self->size;
}

int THClStorage_(elementSize)(THClState *state)
{
  return sizeof(real);
}

void THClStorage_(set)(THClState *state,
		       THClStorage *self, ptrdiff_t index, real value)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  auto queue = state->currentQueue;
}

THCL_API real THClStorage_(get)(THClState *state, const THClStorage*, ptrdiff_t);

THCL_API THClStorage* THClStorage_(new)(THClState *state);
THCL_API THClStorage* THClStorage_(newWithSize)(THClState *state, ptrdiff_t size);
THCL_API THClStorage* THClStorage_(newWithSize1)(THClState *state, real);
THCL_API THClStorage* THClStorage_(newWithSize2)(THClState *state, real, real);
THCL_API THClStorage* THClStorage_(newWithSize3)(THClState *state, real, real, real);
THCL_API THClStorage* THClStorage_(newWithSize4)(THClState *state, real, real, real, real);
THCL_API THClStorage* THClStorage_(newWithMapping)(THClState *state, const char *filename, ptrdiff_t size, int shared);

/* takes ownership of data */
THClStorage* THClStorage_(newWithData)(THClState *state, real *data, ptrdiff_t size)
{
  
}

THClStorage* THClStorage_(newWithAllocator)(
  THClState *state, ptrdiff_t size,
  THClDeviceAllocator* allocator,
  void *allocatorContext)
{
  
}
THClStorage* THClStorage_(newWithDataAndAllocator)(
  THClState *state, real* data, ptrdiff_t size,
  THClDeviceAllocator* allocator,
  void *allocatorContext)
{
  
}

void THClStorage_(setFlag)(THClState *state, THClStorage *storage, const char flag)
{
  storage->flag |= flag;
}
void THClStorage_(clearFlag)(THClState *state, THClStorage *storage, const char flag)
{
  storage->flag &= ~flag;
}
void THClStorage_(retain)(THClState *state, THClStorage *self)
{
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    {
      THAtomicIncrementRef(&self->refcount);
    }
}
void THClStorage_(free)(THClState *state, THClStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (THAtomicDecrementRef(&self->refcount))
    {
      if (self->flag & TH_STORAGE_FREEMEM)
	{
	  (*self->allocator->free)(self->allocatorContext, self->data);
	}
      if(self->flag & TH_STORAGE_VIEW)
	{
	  THClStorage_(free)(state, self->view);
	}
      THFree(self);
    }
}
void THClStorage_(resize)(THClState *state, THClStorage *storage, ptrdiff_t size)
{
  
}
void THClStorage_(fill)(THClState *state, THClStorage *storage, real value)
{
  
}
int THClStorage_(getDevice)(THClState* state, const THClStorage* storage)
{
  
}

#endif
