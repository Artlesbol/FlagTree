// Independent Ascend C reference. All data permutations are vector operations.
// mode 0: split, mode 1: join. One logical row contains Q pairs.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void permute_aiv(
    GM_ADDR x, GM_ADDR z, GM_ADDR y, GM_ADDR indices,
    uint32_t q, uint32_t batch, uint32_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<TPosition::VECCALC> input, output, index;
  pipe.InitBuffer(input, 2 * q * sizeof(float));
  pipe.InitBuffer(output, 2 * q * sizeof(float));
  pipe.InitBuffer(index, 2 * q * sizeof(uint32_t));
  auto in = input.Get<float>();
  auto out = output.Get<float>();
  auto idx = index.Get<uint32_t>();
  GlobalTensor<float> gx, gz, gy;
  GlobalTensor<uint32_t> gi;
  gx.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
  gz.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(z));
  gy.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y));
  gi.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
  // Join uses an immutable vector index table; its HBM read is timed too.
  // This is a reference to improve on, not a claim of hardware peak throughput.
  if (mode == 1) {
    DataCopy(idx, gi, 2 * q);
    pipe_barrier(PIPE_ALL);
  }
  for (uint32_t row = GetBlockIdx(); row < batch; row += GetBlockNum()) {
    if (mode == 0) {
      DataCopy(in, gx[row * 2 * q], 2 * q);
    } else {
      DataCopy(in, gx[row * q], q);
      DataCopy(in[q], gz[row * q], q);
    }
    pipe_barrier(PIPE_ALL);
    if (mode == 0) {
      uint64_t count = 0;
      GatherMaskParams p{1, static_cast<uint8_t>(2 * q / 64), 8, 0};
      GatherMask(out, in, uint8_t(1), false, 0, p, count);
      GatherMask(out[q], in, uint8_t(2), false, 0, p, count);
    } else {
      Gather(out, in, idx, uint32_t(0), 2 * q);
    }
    pipe_barrier(PIPE_ALL);
    DataCopy(gy[row * 2 * q], out, 2 * q);
    pipe_barrier(PIPE_ALL);
  }
}
