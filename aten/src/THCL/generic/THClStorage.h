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


THCL_API real* THClStorage_(data)(const THClStorage*);
THCL_API ptrdiff_t THClStorage_(size)(const THClStorage*);
THCL_API int THClStorage_(elementSize)(void);

/* slow access -- checks everything */
THCL_API void THClStorage_(set)(THClStorage*, ptrdiff_t, real);
THCL_API real THClStorage_(get)(const THClStorage*, ptrdiff_t);

THCL_API THClStorage* THClStorage_(new)(void);
THCL_API THClStorage* THClStorage_(newWithSize)(ptrdiff_t size);
THCL_API THClStorage* THClStorage_(newWithSize1)(real);
THCL_API THClStorage* THClStorage_(newWithSize2)(real, real);
THCL_API THClStorage* THClStorage_(newWithSize3)(real, real, real);
THCL_API THClStorage* THClStorage_(newWithSize4)(real, real, real, real);
THCL_API THClStorage* THClStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int shared);

/* takes ownership of data */
THCL_API THClStorage* THClStorage_(newWithData)(real *data, ptrdiff_t size);

THCL_API THClStorage* THClStorage_(newWithAllocator)
  (ptrdiff_t size,
   THClDeviceAllocator* allocator,
   void *allocatorContext);
THCL_API THClStorage* THClStorage_(newWithDataAndAllocator)
  (real* data, ptrdiff_t size,
   THClDeviceAllocator* allocator,
   void *allocatorContext);

THCL_API void THClStorage_(setFlag)(THClStorage *storage, const char flag);
THCL_API void THClStorage_(clearFlag)(THClStorage *storage, const char flag);
THCL_API void THClStorage_(retain)(THClStorage *storage);

THCL_API void THClStorage_(free)(THClStorage *storage);
THCL_API void THClStorage_(resize)(THClStorage *storage, ptrdiff_t size);
THCL_API void THClStorage_(fill)(THClStorage *storage, real value);

THCL_API int THClStorage_(getDevice)(const THClStorage* storage);

#endif
