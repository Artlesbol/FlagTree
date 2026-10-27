// Device ABI tests use the actual packaged bitcode, not a second algorithm.
#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;

template <typename T> struct ContractMemref {
  T *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
#define DECLARE_WORD(SUFFIX, T) \
  extern "C" __aicore__ void _mlir_ciface_custom_flagtree_stride_gather_##SUFFIX( \
    ContractMemref<__ubuf__ T>*, ContractMemref<__ubuf__ int32_t>*, \
    ContractMemref<__ubuf__ T>*, ContractMemref<__ubuf__ int32_t>*); \
  extern "C" __aicore__ void _mlir_ciface_custom_flagtree_split_##SUFFIX( \
    ContractMemref<__ubuf__ T>*, ContractMemref<__ubuf__ T>*, \
    ContractMemref<__ubuf__ T>*, ContractMemref<__ubuf__ int32_t>*);
DECLARE_WORD(int16, int16_t)
DECLARE_WORD(int32, int32_t)
extern "C" __aicore__ void _mlir_ciface_custom_flagtree_byte_to_word(
    ContractMemref<__ubuf__ int8_t>*, ContractMemref<__ubuf__ int16_t>*,
    ContractMemref<__ubuf__ half>*);
extern "C" __aicore__ void _mlir_ciface_custom_flagtree_word_to_byte(
    ContractMemref<__ubuf__ int16_t>*, ContractMemref<__ubuf__ int8_t>*,
    ContractMemref<__ubuf__ half>*);

constexpr uint32_t STORAGE = 512;
template <typename T>
__aicore__ inline ContractMemref<__ubuf__ T> descriptor(LocalTensor<T> tensor,
    uint32_t n, uint32_t stride, uint32_t offset) {
  auto ptr = reinterpret_cast<__ubuf__ T *>(tensor.GetPhyAddr());
  return {ptr, ptr, offset, {n}, {stride}};
}

template <typename T>
__aicore__ inline void word_contract(GM_ADDR x, GM_ADDR indices,
    GM_ADDR a, GM_ADDR b, GlobalTensor<uint32_t> config, uint32_t mode) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> inBuffer, aBuffer, bBuffer, iBuffer, scratchBuffer;
  pipe.InitBuffer(inBuffer, STORAGE*sizeof(T));
  pipe.InitBuffer(aBuffer, STORAGE*sizeof(T));
  pipe.InitBuffer(bBuffer, STORAGE*sizeof(T));
  pipe.InitBuffer(iBuffer, STORAGE*sizeof(int32_t));
  pipe.InitBuffer(scratchBuffer, STORAGE*sizeof(int32_t));
  auto in = inBuffer.Get<T>(), outA = aBuffer.Get<T>(), outB = bBuffer.Get<T>();
  auto idx = iBuffer.Get<int32_t>(), scratch = scratchBuffer.Get<int32_t>();
  GlobalTensor<T> gx, ga, gb;
  GlobalTensor<int32_t> gi;
  gx.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
  ga.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a));
  gb.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(b));
  gi.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices));
  uint32_t n = config.GetValue(0), repeats = config.GetValue(1);
  auto src = descriptor(in, 2*n, config.GetValue(2), config.GetValue(3));
  auto dstA = descriptor(outA, n, config.GetValue(4), config.GetValue(5));
  auto dstB = descriptor(outB, n, config.GetValue(6), config.GetValue(7));
  auto index = descriptor(idx, n, config.GetValue(8), config.GetValue(9));
  auto temp = descriptor(scratch, n, config.GetValue(10), config.GetValue(11));
  for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
    DataCopy(in, gx, STORAGE);
    DataCopy(idx, gi, STORAGE);
    Duplicate(outA, T(-91), STORAGE);
    Duplicate(outB, T(-91), STORAGE);
    Duplicate(scratch, int32_t(-177), STORAGE);
    pipe_barrier(PIPE_ALL);
    if constexpr (sizeof(T) == 2) {
      if (mode == 0)
        _mlir_ciface_custom_flagtree_stride_gather_int16(&src, &index, &dstA, &temp);
      else
        _mlir_ciface_custom_flagtree_split_int16(&src, &dstA, &dstB, &temp);
    } else {
      if (mode == 0)
        _mlir_ciface_custom_flagtree_stride_gather_int32(&src, &index, &dstA, &temp);
      else
        _mlir_ciface_custom_flagtree_split_int32(&src, &dstA, &dstB, &temp);
    }
    pipe_barrier(PIPE_ALL);
  }
  DataCopy(ga, outA, STORAGE);
  DataCopy(gb, outB, STORAGE);
  pipe_barrier(PIPE_ALL);
}

template <bool EXTEND>
__aicore__ inline void cast_contract(GM_ADDR x, GM_ADDR a,
    GlobalTensor<uint32_t> config) {
  using Src = typename std::conditional<EXTEND, int8_t, int16_t>::type;
  using Dst = typename std::conditional<EXTEND, int16_t, int8_t>::type;
  TPipe pipe;
  TBuf<TPosition::VECCALC> inputBuffer, outputBuffer, tempBuffer;
  pipe.InitBuffer(inputBuffer, STORAGE*sizeof(Src));
  pipe.InitBuffer(outputBuffer, STORAGE*sizeof(Dst));
  pipe.InitBuffer(tempBuffer, STORAGE*sizeof(half));
  auto input = inputBuffer.Get<Src>();
  auto output = outputBuffer.Get<Dst>();
  auto temp = tempBuffer.Get<half>();
  GlobalTensor<Src> gx;
  GlobalTensor<Dst> ga;
  gx.SetGlobalBuffer(reinterpret_cast<__gm__ Src *>(x));
  ga.SetGlobalBuffer(reinterpret_cast<__gm__ Dst *>(a));
  auto src = descriptor(input, config.GetValue(0), config.GetValue(2), config.GetValue(3));
  auto dst = descriptor(output, config.GetValue(0), config.GetValue(4), config.GetValue(5));
  auto scratch = descriptor(temp, config.GetValue(0), config.GetValue(10), config.GetValue(11));
  for (uint32_t repeat = 0; repeat < config.GetValue(1); ++repeat) {
    DataCopy(input, gx, STORAGE);
    if constexpr (sizeof(Dst) == 1) {
      auto poison = outputBuffer.Get<int16_t>();
      Duplicate(poison, int16_t(-23131), STORAGE/2); // 0xa5a5: both bytes -91
    } else {
      Duplicate(output, Dst(-91), STORAGE);
    }
    pipe_barrier(PIPE_ALL);
    if constexpr (EXTEND)
      _mlir_ciface_custom_flagtree_byte_to_word(&src, &dst, &scratch);
    else
      _mlir_ciface_custom_flagtree_word_to_byte(&src, &dst, &scratch);
    pipe_barrier(PIPE_ALL);
  }
  DataCopy(ga, output, STORAGE);
  pipe_barrier(PIPE_ALL);
}

extern "C" __global__ __aicore__ void ub_contract_aiv(GM_ADDR x, GM_ADDR indices,
    GM_ADDR a, GM_ADDR b, GM_ADDR cfg, uint32_t mode, uint32_t wordBytes) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if !defined(__DAV_C220_CUBE__)
  GlobalTensor<uint32_t> config;
  config.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(cfg));
  if (mode == 2) cast_contract<true>(x, a, config);
  else if (mode == 3) cast_contract<false>(x, a, config);
  else if (wordBytes == 2) word_contract<int16_t>(x, indices, a, b, config, mode);
  else word_contract<int32_t>(x, indices, a, b, config, mode);
#endif
}
