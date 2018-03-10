#ifndef THCL_GENERIC_FILE
#define THCL_GENERIC_FILE "generic/THClStorage.h"
#else

#define TH_STORAGE_REFCOUNTED 1
#define TH_STORAGE_RESIZABLE  2
#define TH_STORAGE_FREEMEM    4

typedef struct THClStorage
{
  real *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THClDeviceAllocator *allocator;
  void *allocatorContext;
  struct THClStorage *view;
  int device;
} THClStorage;


THCL_API real* THClStorage_(data)(THClState *state, const THClStorage*);
THCL_API ptrdiff_t THClStorage_(size)(THClState *state, const THClStorage*);
THCL_API int THClStorage_(elementSize)(THClState *state);

/* slow access -- checks everything */
THCL_API void THClStorage_(set)(THClState *state, THClStorage*, ptrdiff_t, real);
THCL_API real THClStorage_(get)(THClState *state, const THClStorage*, ptrdiff_t);

THCL_API THClStorage* THClStorage_(new)(THClState *state);
THCL_API THClStorage* THClStorage_(newWithSize)(THClState *state, ptrdiff_t size);
THCL_API THClStorage* THClStorage_(newWithSize1)(THClState *state, real);
THCL_API THClStorage* THClStorage_(newWithSize2)(THClState *state, real, real);
THCL_API THClStorage* THClStorage_(newWithSize3)(THClState *state, real, real, real);
THCL_API THClStorage* THClStorage_(newWithSize4)(THClState *state, real, real, real, real);
THCL_API THClStorage* THClStorage_(newWithMapping)(THClState *state, const char *filename, ptrdiff_t size, int shared);

/* takes ownership of data */
THCL_API THClStorage* THClStorage_(newWithData)(THClState *state, real *data, ptrdiff_t size);

THCL_API THClStorage* THClStorage_(newWithAllocator)(
  THClState *state, ptrdiff_t size,
  THClDeviceAllocator* allocator,
  void *allocatorContext);
THCL_API THClStorage* THClStorage_(newWithDataAndAllocator)(
  THClState *state, real* data, ptrdiff_t size,
  THClDeviceAllocator* allocator,
  void *allocatorContext);

THCL_API void THClStorage_(setFlag)(THClState *state, THClStorage *storage, const char flag);
THCL_API void THClStorage_(clearFlag)(THClState *state, THClStorage *storage, const char flag);
THCL_API void THClStorage_(retain)(THClState *state, THClStorage *storage);

THCL_API void THClStorage_(free)(THClState *state, THClStorage *storage);
THCL_API void THClStorage_(resize)(THClState *state, THClStorage *storage, ptrdiff_t size);
THCL_API void THClStorage_(fill)(THClState *state, THClStorage *storage, real value);

THCL_API int THClStorage_(getDevice)(THClState* state, const THClStorage* storage);

#endif
