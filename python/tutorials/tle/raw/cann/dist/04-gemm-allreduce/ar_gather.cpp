// Copyright 2026-     FlagOS Contributors
// Thin raw RMA gather only: barrier + peer DataCopy into gathered[pe * n +
// ...]. Triton owns the fp32 reduce (see ar_reduce_kernel). Segmented hybrid
// path.

#include "Utils.h"
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

namespace {
constexpr uint32_t kChunk = 4096;
constexpr int64_t kAlign = 16;
} // namespace

extern "C" {

// Operands (tle_raw.call order with output_indices=[8]):
//   inputs:  symm, gathered, symm_off, gath_off, n, world, pid, nblocks
//   outputs: dummy (UB)
__aiv__ void _mlir_ciface_cannshmem_ar_gather(
    memref_t<__gm__ half, 1> *symm, memref_t<__gm__ half, 1> *gathered,
    int64_t symm_off, int64_t gath_off, int64_t n, int64_t world, int64_t pid,
    int64_t nblocks, memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  __gm__ half *local = symm->aligned + symm->offset;
  __gm__ half *gout = gathered->aligned + gathered->offset;
  const int me = aclshmem_my_pe();
  const int npes = aclshmem_n_pes();
  if (world > 0 && world < npes) {
    // honor host WORLD constexpr; still bound by runtime npes
  }
  int n_pe = npes;
  if (nblocks < 1) {
    nblocks = 1;
  }
  if (pid < 0 || pid >= nblocks) {
    return;
  }

  int64_t n_align = (n + kAlign - 1) & ~(kAlign - 1);
  int64_t slice = ((n_align / nblocks) + kAlign - 1) & ~(kAlign - 1);
  int64_t my_off = pid * slice;
  int64_t my_n = 0;
  if (my_off < n) {
    my_n = slice;
    if (my_off + my_n > n) {
      my_n = n - my_off;
    }
  }

  TPipe pipe;
  TBuf<> tbuf;
  pipe.InitBuffer(tbuf, kChunk * sizeof(half));
  LocalTensor<half> tmp = tbuf.Get<half>(kChunk);

  aclshmemx_barrier_all_vec();

  for (int pe = 0; pe < n_pe; ++pe) {
    __gm__ half *rbase =
        (pe == me) ? local
                   : (__gm__ half *)aclshmem_ptr((__gm__ void *)local, pe);
    for (int64_t off = 0; off < my_n; off += kChunk) {
      uint32_t len = static_cast<uint32_t>(
          (my_n - off) < static_cast<int64_t>(kChunk) ? (my_n - off) : kChunk);
      uint32_t len_align = (len + 15u) & ~15u;
      int64_t g_off = my_off + off;

      GlobalTensor<half> g_src;
      g_src.SetGlobalBuffer(rbase + symm_off + g_off);
      DataCopy(tmp, g_src, len_align);
      SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);

      GlobalTensor<half> g_dst;
      g_dst.SetGlobalBuffer(gout + gath_off + static_cast<int64_t>(pe) * n +
                            g_off);
      DataCopy(g_dst, tmp, len_align);
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
  }

  aclshmemx_barrier_all_vec();
}

} // extern "C"
