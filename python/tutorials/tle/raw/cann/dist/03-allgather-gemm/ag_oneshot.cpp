// Copyright 2026-     FlagOS Contributors
// Oneshot AllGather via cannshmem GM↔GM RMA (NOT UB-staged DataCopy).
//
// Previous version pulled peers through UB tiles — that is the slow path.
// Correct single-node shortcut (NV getmem / P2P equivalent on Ascend):
//   aclshmem_getmem(dst, src_symm, nbytes, peer)
// which dispatches MTE/SDMA/UDMA/RDMA under the hood (see shmem_device_rma.h).
//
// Host already published A_local into ``symm`` and workspace[me].
// This kernel only pulls remote PEs into workspace[peer * n_local :].

#include "Utils.h"
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

extern "C" {

// inputs: symm, workspace, n_local, pid, nblocks | output: dummy
// Only pid==0 issues RMA (one full getmem per peer); other AIVs are no-ops
// so we do not fight "Concurrent RMA to same PE" restrictions.
__aiv__ void
_mlir_ciface_cannshmem_ag_oneshot(memref_t<__gm__ half, 1> *symm,
                                  memref_t<__gm__ half, 1> *workspace,
                                  int64_t n_local, int64_t pid, int64_t nblocks,
                                  memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  (void)nblocks;
  __gm__ half *local = symm->aligned + symm->offset;
  __gm__ half *ws = workspace->aligned + workspace->offset;
  const int me = aclshmem_my_pe();
  const int npes = aclshmem_n_pes();
  if (pid != 0 || n_local <= 0) {
    return;
  }

  // nbytes for getmem (OpenSHMEM-style getmem takes byte count).
  const uint32_t nbytes = static_cast<uint32_t>(n_local * static_cast<int64_t>(sizeof(half)));

  aclshmemx_barrier_all_vec();

  for (int pe = 0; pe < npes; ++pe) {
    if (pe == me) {
      continue;
    }
    __gm__ void *dst = (__gm__ void *)(ws + static_cast<int64_t>(pe) * n_local);
    // src is the symmetric address of A_local on ``pe``.
    aclshmem_getmem(dst, (__gm__ void *)local, nbytes, pe);
  }

  aclshmemx_barrier_all_vec();
}

} // extern "C"
