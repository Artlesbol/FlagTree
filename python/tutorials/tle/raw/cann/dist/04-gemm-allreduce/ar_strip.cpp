// Copyright 2026-     FlagOS Contributors
// Strip-granular oneshot AR with P2P signals (Triton-dist notify / CANN tile
// Commit style) — NO aclshmemx_barrier_all_vec (that deadlocks under dual-stream
// pipelining when not all AIVs enter together).
//
// ready layout: int32[num_strips * world]; producer ``me`` does
//   signal_op(ready + strip * world + me, epoch, SET, pe) for every pe.
// Consumer waits locally on ready[strip * world + p] for all p, then
// peer-get + sum the strip's contiguous half range [elem_off, elem_off+elem_n).

#include "Utils.h"
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

namespace {
constexpr uint32_t kChunk = 4096;
constexpr int64_t kAlign = 16;
} // namespace

extern "C" {

// Mark strip ready: SET ready[strip*world+me]=epoch on every PE (notify).
// inputs: ready, strip, epoch, world, pid | output: dummy
__aiv__ void
_mlir_ciface_cannshmem_mark_strip(memref_t<__gm__ int32_t, 1> *ready,
                                  int64_t strip, int64_t epoch, int64_t world,
                                  int64_t pid,
                                  memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  if (pid != 0) {
    return;
  }
  __gm__ int32_t *base = ready->aligned + ready->offset;
  const int me = aclshmem_my_pe();
  const int npes = aclshmem_n_pes();
  const int nworld =
      (world > 0 && world <= npes) ? static_cast<int>(world) : npes;
  const int32_t sig = static_cast<int32_t>(epoch);
  __gm__ int32_t *slot = base + strip * nworld + me;
  aclshmem_fence();
  for (int pe = 0; pe < nworld; ++pe) {
    aclshmemx_signal_op(slot, sig, ACLSHMEM_SIGNAL_SET, pe);
  }
}

// Wait only: block until ready[strip*world+p] >= epoch for all p.
// inputs: ready, strip, epoch, world, pid | output: dummy
__aiv__ void
_mlir_ciface_cannshmem_wait_strip(memref_t<__gm__ int32_t, 1> *ready,
                                  int64_t strip, int64_t epoch, int64_t world,
                                  int64_t pid,
                                  memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  (void)pid;
  __gm__ int32_t *rdy = ready->aligned + ready->offset;
  const int npes = aclshmem_n_pes();
  const int nworld =
      (world > 0 && world <= npes) ? static_cast<int>(world) : npes;
  const int32_t want = static_cast<int32_t>(epoch);
  for (int p = 0; p < nworld; ++p) {
    aclshmem_signal_wait_until(rdy + strip * nworld + p, ACLSHMEM_CMP_GE, want);
  }
}

// Wait peer strip signals + oneshot reduce for [elem_off, elem_off+elem_n).
// inputs: symm, out, ready, strip, epoch, world, elem_off, elem_n, pid, nblocks
// output: dummy
__aiv__ void
_mlir_ciface_cannshmem_ar_strip(memref_t<__gm__ half, 1> *symm,
                                memref_t<__gm__ half, 1> *out,
                                memref_t<__gm__ int32_t, 1> *ready,
                                int64_t strip, int64_t epoch, int64_t world,
                                int64_t elem_off, int64_t elem_n, int64_t pid,
                                int64_t nblocks,
                                memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  __gm__ half *local = symm->aligned + symm->offset;
  __gm__ half *gout = out->aligned + out->offset;
  __gm__ int32_t *rdy = ready->aligned + ready->offset;
  const int me = aclshmem_my_pe();
  const int npes = aclshmem_n_pes();
  const int nworld =
      (world > 0 && world <= npes) ? static_cast<int>(world) : npes;
  if (nblocks < 1) {
    nblocks = 1;
  }
  if (pid < 0 || pid >= nblocks || elem_n <= 0) {
    return;
  }

  const int32_t want = static_cast<int32_t>(epoch);
  // All AIVs wait so strip data is globally ready before any peer get.
  for (int p = 0; p < nworld; ++p) {
    aclshmem_signal_wait_until(rdy + strip * nworld + p, ACLSHMEM_CMP_GE, want);
  }

  int64_t n_align = (elem_n + kAlign - 1) & ~(kAlign - 1);
  int64_t slice = ((n_align / nblocks) + kAlign - 1) & ~(kAlign - 1);
  int64_t my_off = pid * slice;
  int64_t my_n = 0;
  if (my_off < elem_n) {
    my_n = slice;
    if (my_off + my_n > elem_n) {
      my_n = elem_n - my_off;
    }
  }

  TPipe pipe;
  TBuf<> tbuf;
  pipe.InitBuffer(tbuf, kChunk * sizeof(half) * 2);
  LocalTensor<half> acc = tbuf.Get<half>(kChunk);
  LocalTensor<half> tmp =
      tbuf.GetWithOffset<half>(kChunk, kChunk * sizeof(half));

  for (int64_t off = 0; off < my_n; off += kChunk) {
    uint32_t len = static_cast<uint32_t>(
        (my_n - off) < static_cast<int64_t>(kChunk) ? (my_n - off) : kChunk);
    uint32_t len_align = (len + 15u) & ~15u;
    int64_t g_off = elem_off + my_off + off;

    GlobalTensor<half> g_local;
    g_local.SetGlobalBuffer(local + g_off);
    DataCopy(acc, g_local, len_align);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    for (int pe = 0; pe < nworld; ++pe) {
      if (pe == me) {
        continue;
      }
      __gm__ half *rbase =
          (__gm__ half *)aclshmem_ptr((__gm__ void *)local, pe);
      GlobalTensor<half> g_remote;
      g_remote.SetGlobalBuffer(rbase + g_off);
      DataCopy(tmp, g_remote, len_align);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
      Add(acc, acc, tmp, len_align);
      PipeBarrier<PIPE_V>();
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<half> g_out;
    g_out.SetGlobalBuffer(gout + g_off);
    DataCopy(g_out, acc, len_align);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  }
}

} // extern "C"
