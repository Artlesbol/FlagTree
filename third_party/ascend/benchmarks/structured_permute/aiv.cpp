// Independent Ascend C reference. All data permutations are vector operations.
// mode 0: split, mode 1: join, mode 2: transpose/nested split with 3 outputs.
#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
__aicore__ inline void permute_impl(GM_ADDR x, GM_ADDR z, GM_ADDR y,
                                    GM_ADDR indices, GM_ADDR t, GM_ADDR j,
                                    uint32_t q, uint32_t batch, uint32_t mode) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> input, output, index;
  const uint32_t width = (mode == 2 ? 4 : 2) * q;
  pipe.InitBuffer(input, width * sizeof(T));
  pipe.InitBuffer(output, width * sizeof(T));
  pipe.InitBuffer(index, width * sizeof(uint32_t));
  auto in = input.Get<T>();
  auto out = output.Get<T>();
  auto idx = index.Get<uint32_t>();
  GlobalTensor<T> gx, gz, gy, gt, gj;
  GlobalTensor<uint32_t> gi;
  gx.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
  gz.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(z));
  gy.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
  gt.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(t));
  gj.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(j));
  gi.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
  // Join uses an immutable vector index table; its HBM read is timed too.
  // This is a reference to improve on, not a claim of hardware peak throughput.
  if (mode != 0) {
    DataCopy(idx, gi, width);
    pipe_barrier(PIPE_ALL);
  }
  for (uint32_t row = GetBlockIdx(); row < batch; row += GetBlockNum()) {
    if (mode != 1) {
      DataCopy(in, gx[row * width], width);
    } else {
      DataCopy(in, gx[row * q], q);
      DataCopy(in[q], gz[row * q], q);
    }
    pipe_barrier(PIPE_ALL);
    if (mode == 0) {
      uint64_t count = 0;
      GatherMaskParams p{1, 1, 8, 0};
      GatherMask(out, in, uint8_t(1), true, 2 * q, p, count);
      GatherMask(out[q], in, uint8_t(2), true, 2 * q, p, count);
    } else {
      Gather(out, in, idx, uint32_t(0), width);
    }
    pipe_barrier(PIPE_ALL);
    if (mode == 2) {
      DataCopy(gy[row * width], in, q);
      DataCopy(gy[row * width + q], in[2 * q], q);
      DataCopy(gy[row * width + 2 * q], in[q], q);
      DataCopy(gy[row * width + 3 * q], in[3 * q], q);
      DataCopy(gt[row * width], out, width);
      DataCopy(gj[row * width], out, width);
    } else {
      DataCopy(gy[row * width], out, width);
    }
    pipe_barrier(PIPE_ALL);
  }
}

extern "C" __global__ __aicore__ void
permute_aiv(GM_ADDR x, GM_ADDR z, GM_ADDR y, GM_ADDR indices, GM_ADDR t,
            GM_ADDR j, uint32_t q, uint32_t batch, uint32_t mode,
            uint32_t element_bytes) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if (element_bytes == 2)
    permute_impl<uint16_t>(x, z, y, indices, t, j, q, batch, mode);
  else
    permute_impl<uint32_t>(x, z, y, indices, t, j, q, batch, mode);
}

// Independent matrix-transport reference: preserve aligned UB rows from the
// load through the byte-counted GM store. Arithmetic and permutations are SIMD.
extern "C" __global__ __aicore__ void matrix_transport_aiv(GM_ADDR x, GM_ADDR y,
                                                           GM_ADDR indices,
                                                           uint32_t transpose) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<TPosition::VECCALC> inputBuffer, outputBuffer, indexBuffer;
  uint32_t elements = transpose ? 128 : 8192;
  pipe.InitBuffer(inputBuffer, elements * sizeof(float));
  pipe.InitBuffer(outputBuffer, transpose ? 512 : 32);
  pipe.InitBuffer(indexBuffer, transpose ? 512 : 32);
  auto input = inputBuffer.Get<float>();
  auto output = outputBuffer.Get<float>();
  auto index = indexBuffer.Get<uint32_t>();
  GlobalTensor<float> gx, gy;
  GlobalTensor<uint32_t> gi;
  gx.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
  gy.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y));
  gi.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
  DataCopyPadExtParams<float> padding;
  if (transpose) {
    DataCopy(index, gi, 128);
    pipe_barrier(PIPE_ALL);
  }
  uint32_t programs = transpose ? 56 : 256;
  for (uint32_t program = GetBlockIdx(); program < programs;
       program += GetBlockNum()) {
    if (transpose) {
      uint32_t columns = 221 - program * 4;
      columns = columns < 4 ? columns : 4;
      Duplicate(input, float(0), 128);
      pipe_barrier(PIPE_ALL);
      DataCopyExtParams load(11, columns * 8, (442 - columns * 2) * 4, 0, 0);
      DataCopyPad(input, gx[program * 8], load, padding);
      pipe_barrier(PIPE_ALL);
      Adds(input, input, float(1), 128);
      pipe_barrier(PIPE_ALL);
      Gather(output, input, index, uint32_t(0), 128);
      pipe_barrier(PIPE_ALL);
      DataCopyExtParams store(columns, 22 * 4, 1, (418 - 22) * 4, 0);
      DataCopyPad(gy[program * 4 * 418], output, store);
    } else {
      DataCopyExtParams load(1024, 16, (1024 - 4) * 4, 0, 0);
      DataCopyPad(input, gx[program * 4], load, padding);
      pipe_barrier(PIPE_ALL);
      Adds(input, input, float(1), 8192);
      pipe_barrier(PIPE_ALL);
      DataCopyExtParams store(1024, 16, 0, (1024 - 4) * 4, 0);
      DataCopyPad(gy[program * 4], input, store);
    }
    pipe_barrier(PIPE_ALL);
  }
}
