// Copyright 2026-     FlagOS Contributors
// Tile-granular oneshot AR consumer (NV wait_ready + reduce).
// Row-major (M,N) workspace: tile (tm,tn) covers
//   rows [tm*BM, min((tm+1)*BM, M)), cols [tn*BN, min((tn+1)*BN, N)).
// No collective barrier inside the tile loop (disjoint AIV ownership).

#include "Utils.h"
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

namespace {
constexpr uint32_t kChunk = 2048;
} // namespace

extern "C" {

// inputs: symm, out, ready, epoch, M, N, BM, BN, num_tiles, pid, nblocks
// output: dummy
__aiv__ void
_mlir_ciface_cannshmem_ar_tile(memref_t<__gm__ half, 1> *symm,
                               memref_t<__gm__ half, 1> *out,
                               memref_t<__gm__ int32_t, 1> *ready,
                               int64_t epoch, int64_t M, int64_t N, int64_t BM,
                               int64_t BN, int64_t num_tiles, int64_t pid,
                               int64_t nblocks,
                               memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  __gm__ half *local = symm->aligned + symm->offset;
  __gm__ half *gout = out->aligned + out->offset;
  __gm__ int32_t *rdy = ready->aligned + ready->offset;
  const int me = aclshmem_my_pe();
  const int npes = aclshmem_n_pes();
  if (nblocks < 1) {
    nblocks = 1;
  }
  if (pid < 0 || pid >= nblocks || M <= 0 || N <= 0 || BM <= 0 || BN <= 0 ||
      num_tiles <= 0) {
    return;
  }

  const int32_t want = static_cast<int32_t>(epoch);
  const int64_t num_pid_n = (N + BN - 1) / BN;

  TPipe pipe;
  TBuf<> tbuf;
  pipe.InitBuffer(tbuf, kChunk * sizeof(half) * 2);
  LocalTensor<half> acc = tbuf.Get<half>(kChunk);
  LocalTensor<half> tmp =
      tbuf.GetWithOffset<half>(kChunk, kChunk * sizeof(half));

  for (int64_t tile = pid; tile < num_tiles; tile += nblocks) {
    for (int pe = 0; pe < npes; ++pe) {
      __gm__ int32_t *peer_rdy =
          (__gm__ int32_t *)aclshmem_ptr((__gm__ void *)rdy, pe);
      aclshmem_signal_wait_until(peer_rdy + tile, ACLSHMEM_CMP_GE, want);
    }

    const int64_t tm = tile / num_pid_n;
    const int64_t tn = tile % num_pid_n;
    const int64_t row0 = tm * BM;
    const int64_t col0 = tn * BN;
    const int64_t row1 = (row0 + BM < M) ? (row0 + BM) : M;
    const int64_t col1 = (col0 + BN < N) ? (col0 + BN) : N;
    const int64_t rows = row1 - row0;
    const int64_t cols = col1 - col0;
    if (rows <= 0 || cols <= 0) {
      continue;
    }

    // Reduce row-by-row (cols may be < BN); chunk along columns.
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; c += kChunk) {
        uint32_t len = static_cast<uint32_t>(
            (cols - c) < static_cast<int64_t>(kChunk) ? (cols - c) : kChunk);
        uint32_t len_align = (len + 15u) & ~15u;
        const int64_t g_off = (row0 + r) * N + (col0 + c);

        GlobalTensor<half> g_local;
        g_local.SetGlobalBuffer(local + g_off);
        DataCopy(acc, g_local, len_align);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (int pe = 0; pe < npes; ++pe) {
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
  }
}

} // extern "C"
