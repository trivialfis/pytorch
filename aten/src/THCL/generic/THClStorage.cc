#ifndef THCL_GENERIC_FILE
#define THCL_GENERIC_FILE "generic/THClStorage.cc"
#else

#include <boost/compute/core.hpp>
#include <boost/compute/command_queue.hpp>

extern THClState* THCl_global_state();

real* THClStorage_(data)(const THClStorage *self)
{
  return self->data;
}

ptrdiff_t THClStorage_(size)(const THClStorage *self)
{
  return self->size;
}

int THClStorage_(elementSize)()
{
  return sizeof(real);
}

void THClStorage_(set)(THClStorage *self, ptrdiff_t index, real value)
{
  THClState *state = THCl_global_state();
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  auto queue = state->currentQueue;
}

real THClStorage_(get)(const THClStorage*, ptrdiff_t)
{
  
}

THClStorage* THClStorage_(new)(void)
{
  
}
THClStorage* THClStorage_(newWithSize)(ptrdiff_t size)
{
  
}
THClStorage* THClStorage_(newWithSize1)(real)
{
  
}
THClStorage* THClStorage_(newWithSize2)(real, real)
{
  
}
THClStorage* THClStorage_(newWithSize3)(real, real, real)
{
  
}
THClStorage* THClStorage_(newWithSize4)(real, real, real, real)
{
  
}
THClStorage* THClStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int shared)
{
  
}

/* takes ownership of data */
THClStorage* THClStorage_(newWithData)(real *data, ptrdiff_t size)
{
  
}

THClStorage* THClStorage_(newWithAllocator)(ptrdiff_t size,
					    THClDeviceAllocator* allocator,
					    void *allocatorContext)
{
  
}
THClStorage* THClStorage_(newWithDataAndAllocator)(real* data, ptrdiff_t size,
						   THClDeviceAllocator* allocator,
						   void *allocatorContext)
{
  
}

void THClStorage_(setFlag)(THClStorage *storage, const char flag)
{
  storage->flag |= flag;
}
void THClStorage_(clearFlag)(THClStorage *storage, const char flag)
{
  storage->flag &= ~flag;
}
void THClStorage_(retain)(THClStorage *self)
{
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    {
      THAtomicIncrementRef(&self->refcount);
    }
}
void THClStorage_(free)(THClStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;
  
  // THClState *state = THCl_global_state();
  if (THAtomicDecrementRef(&self->refcount))
    {
      if (self->flag & TH_STORAGE_FREEMEM)
	{
	  (*self->allocator->free)(self->allocatorContext, self->data);
	}
      if(self->flag & TH_STORAGE_VIEW)
	{
	  THClStorage_(free)(self->view);
	}
      THFree(self);
    }
}
void THClStorage_(resize)(THClStorage *storage, ptrdiff_t size)
{
  
}
void THClStorage_(fill)(THClStorage *storage, real value)
{
  
}
int THClStorage_(getDevice)(const THClStorage* storage)
{
  
}

#endif
