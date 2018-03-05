// from lib/THC/THCTensorMasked.cu:

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <boost/compute/core.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/algorithm/reverse.hpp>
#include <boost/compute/algorithm/transform_if.hpp>
#include <boost/compute/lambda.hpp>
#include <boost/compute/iterator/zip_iterator.hpp>

#include "THClTensorMath.h"
#include "THClGeneral.h"
#include "THClBlas.h"
#include "THClTensorCopy.h"
//#include "THClTensorRandom.h"
#include "THClApply.h"
#include "THClReduce.h"

#include <iostream>
using namespace std;

// The largest consecutive integer representable in float32 (2^24)
#define FLOAT32_MAX_CONSECUTIVE_INT 16777216.0f

class TensorMaskedFillOp : public HasOperator2, public HasScalars {
public:
  int getNumScalars() const { return 1; }
  float getScalar( int index ) const { return value; }
  TensorMaskedFillOp(float v) : value(v) {}
  std::string operator2() const {
    return "if( *in1 != 0.0f ) { *out = val1; }";
  }
  float value;
};

class TensorMaskedCopyOp : public HasOperator2, public HasGlobalTensors {
public:
  TensorMaskedCopyOp(THClTensor *src, THClTensor *baseMask, THClTensor *maskPrefixSum) :
    src(src), baseMask(baseMask), maskPrefixSum(maskPrefixSum) {
  }
  int getNumGlobalTensors() const { return 3; }
  THClTensor *getTensor(int index) const {
    if(index == 0){ return src; }
    if(index == 1){ return baseMask; }
    if(index == 2){ return maskPrefixSum; }
    THError("index not recognized %i", index);
    return 0;
  }
  std::string getTensorName(int index) const {
    if(index == 0){ return "src"; }
    if(index == 1){ return "baseMask"; }
    if(index == 2){ return "maskPrefixSum"; }
    THError("index not recognized %i", index);
    return "";
  }
  std::string operator2() const {
    return "if( *in1 != 0.0f ) { *out = src[(int)maskPrefixSum[srcOffset] ]; }";
  }
  THClTensor *src;
  THClTensor *baseMask;
  THClTensor *maskPrefixSum;
};

//struct TensorMaskedCopyOp {
//  TensorMaskedCopyOp(float* s, float* bm, float* ps)
//      : src(s),
//        baseMask(bm),
//        maskPrefixSum(ps) {
//  }

//  /*__device__*/ /*__forceline__*/ void operator()(float* out, float* mask) {
//    // Really mask should be `0` or `1` but we can't propagate errors here.
//    if (*mask != 0.0f) {
//      // We've already checked that this offset is <= 2^24, so this is ok.
//      int srcOffset = (int) (mask - baseMask);
//      *out = src[(int) maskPrefixSum[srcOffset]];
//    }
//  }

//  // Where we are copying from
//  float* src;

//  // The base address of mask so we can calculate offset
//  float* baseMask;

//  // The index we are copying from
//  float* maskPrefixSum;
//};

//class TensorMaskedSelectOp : public HasOperator3, public HasScalars {
//public:
//  int getNumScalars() const { return 1; }
//  string operator3() const {
//    return "if(*out != 0.0f){out[(int)*in1] = *in2; }";
//  }
//  TensorMaskedSelectOp(float* t) : out(t) {}
//  void operator()(float* mask, float* maskPrefixSum, float* in) {
//    // Really mask should be `0` or `1` but we can't propagate errors here.
//    if (*mask != 0.0f) {
//      out[(int) *maskPrefixSum] = *in;
//    }
//  }

//  float* out;
//};

void THClTensor_maskedFill(THClState* state,
                             THClTensor *tensor, THClTensor *mask, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, tensor, mask));
  THArgCheck(THClTensor_nElement(state, tensor) ==
             THClTensor_nElement(state, mask),
             2, "sizes do not match");

  TensorMaskedFillOp op(value);
  if (!THClTensor_pointwiseApply2(state, tensor, mask, &op)) {
    THArgCheck(false, 2, CLTORCH_DIM_WARNING);
  }
}

void THClTensor_maskedCopy(THClState* state,
                             THClTensor *tensor, THClTensor *mask, THClTensor *src)
{
  THAssert(THClTensor_checkGPU(state, 3, tensor, src, mask));
  long maskSize = THClTensor_nElement(state, mask);
  long tensorSize = THClTensor_nElement(state, tensor);
  long srcSize = THClTensor_nElement(state, src);

  // Since we are performing a prefix sum of mask, it cannot exceed
  // the size allowed in consecutive integers in float32
  THArgCheck(maskSize <= (long) FLOAT32_MAX_CONSECUTIVE_INT,
             3, "mask nElements exceeds single-precision float "
             "consecutive integer precision size (2^24)");

  // `mask` and `tensor` must have the same number of elements
  THArgCheck(maskSize == tensorSize, 2,
             "mask and tensor must have the same number of elements");

  THClTensor* contigMask = THClTensor_newContiguous(state, mask);
  long oneElements = (long) THClTensor_sumall(state, contigMask);

  // The number of `1` elements present in the mask must be <= the
  // number of elements available in `src`
  if (oneElements > srcSize) {
    THClTensor_free(state, contigMask);
    THArgCheck(false, 2, "source nElements must be == mask `1` elements");
  }

  // Use a prefix sum to determine the copy locations of the masked elements
  THClTensor* maskPrefixSum = THClTensor_newv2(state, src->storage->device);
  THClTensor_resizeAs(state, maskPrefixSum, contigMask);

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  THClTensor* contigSrc = THClTensor_newContiguous(state, src);

//  TensorAddOp cumOp;
//  THAssert(THClTensor_checkGPU(state, 2, maskData, maskPrefixSumData));
//  THClTensor_scanDim(state, maskPrefixSumData, maskData, dimension, 0.0f, &cumOp);
  // hmmmm: this is scanDim, but what we really need is somsething like: scanAll

//   thrust::device_ptr<float>
//    maskData(THClTensor_data(state, contigMask));
//   thrust::device_ptr<float>
//    maskPrefixSumData(THClTensor_data(state, maskPrefixSum));
//   thrust::exclusive_scan(maskData,
//                         maskData + THClTensor_nElement(state, contigMask),
//                         maskPrefixSumData);
  THError("Not implemented");

  // update `tensor` where `mask` == 1 but pull from `src` at
  // maskPrefixSum
  TensorMaskedCopyOp maskedCopyOp(
     contigSrc, contigMask, maskPrefixSum);
  bool status = THClTensor_pointwiseApply2(
    state, tensor, contigMask, &maskedCopyOp);
  THError("Not implemented");

  THClTensor_free(state, contigSrc);
  THClTensor_free(state, maskPrefixSum);
  THClTensor_free(state, contigMask);

  THArgCheck(status, 2, CLTORCH_DIM_WARNING);

  THError("Not implemented");
}

void THClTensor_maskedSelect(THClState* state,
                              THClTensor *tensor, THClTensor *src, THClTensor *mask)
  {
  THAssert(THClTensor_checkGPU(state, 3, tensor, src, mask));
  THArgCheck(THClTensor_nElement(state, mask) == THClTensor_nElement(state, src),
            2, "sizes do not match");

  THClTensor* contigSrc = THClTensor_newContiguous(state, src);
  THClTensor* contigMask = THClTensor_newContiguous(state, mask);

  int sourceSize = THClTensor_nElement(state, contigMask);

  EasyCL *cl = src->storage->cl;

  boost::compute::context boost_context(*cl->context);
  boost::compute::command_queue boost_queue(*cl->queue);

  boost::compute::buffer boostData(*contigSrc->storage->wrapper->getDeviceArray());
  boost::compute::buffer boostMask(*contigMask->storage->wrapper->getDeviceArray());

  size_t copied_element_count = ::boost::compute::count(
    boost::compute::make_buffer_iterator<float>(boostMask, 0),
    boost::compute::make_buffer_iterator<float>(boostMask, sourceSize),
    1,
    boost_queue
  );
  int totalElements = (int)copied_element_count;

  THClTensor_resize1d(state, tensor, totalElements);
  boost::compute::buffer boostOut(*tensor->storage->wrapper->getDeviceArray());

  transform_if(
    make_zip_iterator(
      boost::make_tuple(
      boost::compute::make_buffer_iterator<float>(boostData, 0),
      boost::compute::make_buffer_iterator<float>(boostMask, 0)
      )
    ),
    make_zip_iterator(
      boost::make_tuple(
      boost::compute::make_buffer_iterator<float>(boostData, sourceSize),
      boost::compute::make_buffer_iterator<float>(boostMask, sourceSize)
      )
    ),
    boost::compute::make_buffer_iterator<float>(boostOut, 0),
    boost::compute::get<0>(), // function that return input value
    boost::compute::lambda::get<1>(boost::compute::_1) == 1, // lambda function that checks if mask is 1
    boost_queue // command queue (boost::compute::command_queue object)
  );
  tensor->storage->wrapper->markDeviceDirty();

  THClTensor_free(state, contigSrc);
  THClTensor_free(state, contigMask);
}

void THClTensor_maskedFillByte(THClState* state, THClTensor *tensor, THByteTensor *mask, float value)
{
 THAssert(THClTensor_checkGPU(state, 1, tensor));
 THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
 const int device = tensor->storage->device;
 THClTensor* maskCl = THClTensor_newWithSize(state, device, maskSize, NULL);
 THLongStorage_free(maskSize);
 THClTensor_copyByte(state, maskCl, mask);
 THClTensor_maskedFill(state, tensor, maskCl, value);
 THClTensor_free(state, maskCl);
}

//void THClTensor_maskedCopyByte(THClState* state, THClTensor *tensor, THByteTensor *mask, THClTensor *src)
//{
//  THAssert(THClTensor_checkGPU(state, 2, tensor, src));
//  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
//  THClTensor* maskCl = THClTensor_newWithSize(state, maskSize, NULL);
//  THLongStorage_free(maskSize);
//  THClTensor_copyByte(state, maskCl, mask);
//  THClTensor_maskedCopy(state, tensor, maskCl, src);
//  THClTensor_free(state, maskCl);
//}

void THClTensor_maskedSelectByte(THClState* state, THClTensor *tensor, THClTensor *src, THByteTensor *mask)
{
 THAssert(THClTensor_checkGPU(state, 2, tensor, src));
 THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
 const int device = src->storage->device;
 THClTensor* maskCl = THClTensor_newWithSize(state, device, maskSize, NULL);
 THLongStorage_free(maskSize);
 THClTensor_copyByte(state, maskCl, mask);
 THClTensor_maskedSelect(state, tensor, src, maskCl);
 THClTensor_free(state, maskCl);
}
