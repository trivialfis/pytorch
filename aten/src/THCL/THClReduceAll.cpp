#include "THClKernels.h"
#include "THClReduceAll.h"
#include "THClTypeParseTraits.h"
#include "THClReduceApplyUtils.h"
#include "THClDeviceUtils.h"
#include "EasyCL.h"
#include "templates/TemplatedKernel.h"
#include "util/StatefulTimer.h"

#include <iostream>
#include <string>
using namespace std;

static std::string getKernelTemplate();

int64 getReduceAllBlockSize(THClState *state, int device) {
  int blockSize = 1024;
  int maxWorkgroupSize = ((easycl::DeviceInfo *)state->deviceInfoByDevice[device])->maxWorkGroupSize;
  if( blockSize > maxWorkgroupSize ) {
    blockSize = maxWorkgroupSize;
  }
  return blockSize;
}

// Cutoff size for two-pass reduction
// #define THCL_TWO_PASS_REDUCTION_SIZE 2048L
// I wonder if this is a function of the block size above?  logically it 
// probably is...

int64 getTwoPassReductionSize(THClState *state, int device) {
  return getReduceAllBlockSize(state, device) * 2;
}

// Perform a two-pass reduction if the tensor is large enough to
// warrant it.
bool isTwoPassReductionSize(THClState *state, int device, int64 elements) {
  return (elements > getTwoPassReductionSize(state, device));
}

int64 getTwoPassBlocks(THClState* state, int device, int64 elements) {
  int64 numBlocks = THClCeilDiv(elements, getReduceAllBlockSize(state, device));

  // We can only have as many blocks as there is scratch space
  size_t scratchSpace =
    THClState_getDeviceScratchSpaceSize(state, device) / sizeof(float);
  THAssert(scratchSpace > 0);

  if (numBlocks > (int64)scratchSpace) {
    numBlocks = scratchSpace;
  }

  return numBlocks;
}

// Get the block/grid size that we want
void getPass1ReduceBlockGrid(THClState* state, int device, int64 elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(getTwoPassBlocks(state, device, elements));
  block = dim3(getReduceAllBlockSize(state, device));
}

void getPass2ReduceBlockGrid(THClState* state, int device, int64 elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(1);
  // We only need as many threads as there were blocks originally
  block = dim3(getTwoPassBlocks(state, device, elements));
}

void getSinglePassReduceBlockGrid(THClState *state, int device, int64 elements,
                                         dim3& grid, dim3& block) {
  grid = dim3(1);
  block = dim3(getReduceAllBlockSize(state, device));
}

template< typename IndexType >
void kernelLaunch_THClTensor_reduceAllPass1(
                     THClState* state,
                     dim3 &grid, dim3 &block, size_t smemSize,
                     int ADims,
                     const TensorInfo<IndexType> & in,
                     int64 totalElements,
                     float init,
                     const HasOperator2 *modifyOp,
                     const HasOperator3 *reduceOp,
                     CLWrapper* scratch
    ){
  StatefulTimer::timeCheck("ReduceAllPass1 START");
  std::string uniqueName = "THClTensor_reduceAllPass1_" + easycl::toString(ADims) + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  EasyCL *cl = scratch->getCl();
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("ReduceAllPass1 1aa");
  } else {
    std::vector< int > dims;
    if( ADims >= 0 ) {
      dims.push_back(ADims);
    }
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder
      .set("include_THClDeviceUtils", THClDeviceUtils_getKernelTemplate())
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("dims", dims)
      .set("dim1", ADims)
      .set("defreduceblock", 1)
      .set("modify_operation", modifyOp->operator2())
      .set("reduce_operation", reduceOp->operator3())
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("IndexType", TypeParseTraits<IndexType>::name)
    ;
    kernel = kernelBuilder.buildKernel( uniqueName, "THClReduceAll.cl", getKernelTemplate(), "THClTensor_reduceAllPass1" );
  }

  THClKernels k(state, kernel);
  k.in(in);
  k.in((int)totalElements);
  k.in(init);
  k.out(scratch);
  k.localFloats(smemSize / sizeof(float));
  k.run(grid, block);
  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("ReduceAllPass1 END");
}

template< typename IndexType >
void kernelLaunch_THClTensor_reduceAllPass2(
                     THClState* state,
                     dim3 &grid, dim3 &block, size_t smemSize,
                     int numPass1Blocks,
                     float init,
                     const HasOperator3 *reduceOp,
                     CLWrapper *scratch,
                     CLWrapper* devOut
    ){
  StatefulTimer::timeCheck("ReduceAllPass2 START");
  std::string uniqueName = "THClTensor_reduceAllPass2_" + reduceOp->operator3();
  EasyCL *cl = scratch->getCl();
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("ReduceAllPass2 1aa");
  } else {
    TemplatedKernel kernelBuilder(cl);
    std::vector< int > dims;
    kernelBuilder
      .set("include_THClDeviceUtils", THClDeviceUtils_getKernelTemplate())
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("dims", dims)
      .set("dim1", -2)
      .set("defreduceblock", 1)
      .set("reduce_operation", reduceOp->operator3())
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("IndexType", TypeParseTraits<IndexType>::name)
    ;

    kernel = kernelBuilder.buildKernel( uniqueName, "THClReduceAll.cl", getKernelTemplate(), "THClTensor_reduceAllPass2" );
  }

  THClKernels k(state, kernel);
  k.in(numPass1Blocks);
  k.in(init);
  k.in(scratch);
  k.out(devOut);
  k.localFloats(smemSize / sizeof(float));
  k.run(grid, block);

  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("ReduceAllPass2 End");
}

template< typename IndexType >
void kernelLaunch_THClTensor_reduceAll(
                     THClState* state,
                     dim3 &grid, dim3 &block, size_t smemSize,
                     int ADims,
                     const TensorInfo<IndexType> &in,
                     int64 totalElements,
                     float init,
                     const HasOperator2 *modifyOp,
                     const HasOperator3 *reduceOp,
                     CLWrapper* devOut
    ){
  StatefulTimer::timeCheck("ReduceAll START");
  std::string uniqueName = "THClTensor_reduceAll_" + easycl::toString(ADims) + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  EasyCL *cl = devOut->getCl();
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("ReduceAllPass2 1aa");
  } else {
    std::vector< int > dims;
    if( ADims >= 0 ) {
      dims.push_back(ADims);
    }
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder
      .set("include_THClDeviceUtils", THClDeviceUtils_getKernelTemplate())
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("dims", dims)
      .set("dim1", ADims)
      .set("defreduceblock", 1)
      .set("modify_operation", modifyOp->operator2())
      .set("reduce_operation", reduceOp->operator3())
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("IndexType", TypeParseTraits<IndexType>::name)
    ;

    kernel = kernelBuilder.buildKernel( uniqueName, "THClReduceAll.cl", getKernelTemplate(), "THClTensor_reduceAll" );
  }

  THClKernels k(state, kernel);
  k.in(in);
  k.in((int)totalElements);
  k.in(init);
  k.out(devOut);
  k.localFloats(smemSize / sizeof(float));
  k.run(grid, block);
  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("ReduceAll END");
}

template <typename IndexType>
void callReduceAll(THClState* state,
                   const int device,
                   int ADims,
                   const TensorInfo<IndexType>& in,
                   int64 totalElements,
                   float init,
                   const HasOperator2 *modifyOp,
                   const HasOperator3 *reduceOp,
                   CLWrapper* devOut) {
  dim3 grid;
  dim3 block;

//  TensorInfoCl inCl(in);
  if (isTwoPassReductionSize(state, device, totalElements)) {
    getPass1ReduceBlockGrid(state, device, totalElements, grid, block);
    size_t smemSize = block.x() * sizeof(float);

    kernelLaunch_THClTensor_reduceAllPass1<IndexType>(
        state,
        grid, block, smemSize,
        ADims,
          in,
           (IndexType) totalElements, init, modifyOp, reduceOp,
        THClState_getDeviceScratchSpace(state, device, 0)->wrapper);

    int numPass1Blocks = grid.x();
    getPass2ReduceBlockGrid(state, device, totalElements, grid, block);
    smemSize = block.x() * sizeof(float);

    kernelLaunch_THClTensor_reduceAllPass2<IndexType>(
        state,
        grid, block, smemSize,
        numPass1Blocks, init, reduceOp,
        THClState_getDeviceScratchSpace(state, device, 0)->wrapper,
        devOut);
  } else {
    getSinglePassReduceBlockGrid(state, device, totalElements, grid, block);
    size_t smemSize = block.x() * sizeof(float);

    kernelLaunch_THClTensor_reduceAll<IndexType>(
        state,
        grid, block, smemSize,
        ADims, 
        in,
        (IndexType) totalElements, init, modifyOp, reduceOp, devOut);
  }
}

// Reduces the entire tensor to one floating-point value. `out` points
// to host-resident memory.
bool THClTensor_reduceAll(THClState* state,
                            THClTensor* in,
                            const HasOperator2 *modifyOp,
                            const HasOperator3 *reduceOp,
                            float init,
                            CLWrapper *res) {
  int64 inElements = THClTensor_nElement(state, in);

  if (THClTensor_nDimension(state, in) > MAX_CLTORCH_DIMS) {
    return false;
  }

  if (THClTensor_nDimension(state, in) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const int device = in->device;

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.

  if (THCL_canUse32BitIndexMath(state, in)) {
    TensorInfo<uint32> inInfo(state, in);
    int IN = inInfo.isContiguous() ? -2 : inInfo.dims;
    callReduceAll<uint32>(
      state, device, IN, inInfo, inElements, init, modifyOp, reduceOp, res);
  } else {
    TensorInfo<uint64> inInfo(state, in);
    int IN = inInfo.isContiguous() ? -2 : inInfo.dims;
    callReduceAll<uint64>(
      state, device, IN, inInfo, inElements, init, modifyOp, reduceOp, res);
  }
  return true;
}

std::string getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClReduceAll.cl" )
  // ]]]
  // generated using cog, from THClReduceAll.cl:
  const char * kernelSource =  
  "{{include_THClDeviceUtils}}\n" 
  "\n" 
  "static inline float modifyOp(float _in1) {\n" 
  "  float _out;\n" 
  "  float *in1 = &_in1;\n" 
  "  float *out = &_out;\n" 
  "  {{modify_operation}};\n" 
  "  return _out;\n" 
  "}\n" 
  "\n" 
  "static inline float reduceOp(float _in1, float _in2) {\n" 
  "  // I guess the compiler can sort this stuff out :-P\n" 
  "  float _out;\n" 
  "  float *in1 = &_in1;\n" 
  "  float *in2 = &_in2;\n" 
  "  float *out = &_out;\n" 
  "  {{reduce_operation}};\n" 
  "  return _out;\n" 
  "}\n" 
  "\n" 
  "{{include_THClReduceApplyUtils}}\n" 
  "\n" 
  "// Kernel that handles an entire reduction of a tensor in one pass\n" 
  "kernel void\n" 
  "THClTensor_reduceAll(global TensorInfoCl *in_info,\n" 
  "                     global float *in_data,\n" 
  "                     {{IndexType}} totalElements,\n" 
  "                     float init,\n" 
  "                     global float* out,\n" 
  "                     local float *smem) {\n" 
  "  // With a block-wide stride, have each thread perform its own reduction.\n" 
  "  float r = init;\n" 
  "  for ({{IndexType}} i = get_local_id(0); i < totalElements; i += get_local_size(0)) {\n" 
  "    const {{IndexType}} inOffset = IndexToOffset_{{1000 + dim1}}_get(i, &in_info[0]);\n" 
  "    r = reduceOp(r, modifyOp(in_data[inOffset]));\n" 
  "  }\n" 
  "\n" 
  "  // Reduce within the block\n" 
  "  r = reduceBlock(smem, get_local_size(0), r, init);\n" 
  "\n" 
  "  if(get_local_id(0) == 0) {\n" 
  "    // Write out reduced value\n" 
  "    out[0] = r;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "static inline {{IndexType}} getStartIndex({{IndexType}} totalSize) {\n" 
  "  {{IndexType}} sizePerBlock = THClCeilDiv(totalSize, ({{IndexType}}) get_num_groups(0));\n" 
  "  return get_group_id(0) * sizePerBlock;\n" 
  "}\n" 
  "\n" 
  "static inline {{IndexType}} getEndIndex({{IndexType}} totalSize) {\n" 
  "  {{IndexType}} sizePerBlock = THClCeilDiv(totalSize, ({{IndexType}}) get_num_groups(0));\n" 
  "  return min(({{IndexType}}) ((get_group_id(0) + 1) * sizePerBlock), totalSize);\n" 
  "}\n" 
  "\n" 
  "// Kernel that handles an entire reduction of a tensor in two passes\n" 
  "kernel void\n" 
  "THClTensor_reduceAllPass1(global TensorInfoCl *in_info,\n" 
  "                          global float *in_data,\n" 
  "                          {{IndexType}} totalElements,\n" 
  "                          float init,\n" 
  "                          global float* scratchSpace,\n" 
  "                          local float *smem) {\n" 
  "  const {{IndexType}} startIndex = getStartIndex(totalElements);\n" 
  "  const {{IndexType}} endIndex = getEndIndex(totalElements);\n" 
  "\n" 
  "  // With a block-wide stride, have each thread perform its own reduction.\n" 
  "  float r = init;\n" 
  "  for ({{IndexType}} i = startIndex + get_local_id(0); i < endIndex; i += get_local_size(0)) {\n" 
  "    const {{IndexType}} inOffset = IndexToOffset_{{1000 + dim1}}_get(i, &in_info[0]);\n" 
  "    r = reduceOp(r, modifyOp(in_data[inOffset]));\n" 
  "  }\n" 
  "\n" 
  "  // Reduce within the block\n" 
  "  r = reduceBlock(smem, get_local_size(0), r, init);\n" 
  "\n" 
  "  if ((int)get_local_id(0) == 0) {\n" 
  "    // Write out block-wide reduced value\n" 
  "    scratchSpace[get_group_id(0)] = r;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "kernel void THClTensor_reduceAllPass2(int numPass1Blocks,\n" 
  "                            float init,\n" 
  "                            global float* scratchSpace,\n" 
  "                            global float* out,\n" 
  "                            local float *smem) {\n" 
  "  float r = init;\n" 
  "  if ((int)get_local_id(0) < numPass1Blocks) {\n" 
  "    r = scratchSpace[get_local_id(0)];\n" 
  "  }\n" 
  "\n" 
  "  // Reduce within the block\n" 
  "  r = reduceBlock(smem, numPass1Blocks, r, init);\n" 
  "\n" 
  "  if((int)get_local_id(0) == 0) {\n" 
  "    out[0] = r;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

