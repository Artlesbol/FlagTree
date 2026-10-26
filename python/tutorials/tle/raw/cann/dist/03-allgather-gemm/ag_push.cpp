// Copyright 2026-     FlagOS Contributors
// Full-mesh AG push (NV / TDA / ops-transformer FullMesh style):
//   Each PE putmem's A_local into every peer's workspace[me * n_local :].
//   Uses GM↔GM aclshmem_putmem (NOT UB DataCopy).
// Host already wrote workspace[me] locally; this kernel only pushes to remotes.
// pid==0 issues puts; other AIVs no-op (avoid concurrent RMA to same PE).

#include "Utils.h"
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

extern "C" {

// inputs: a_local, workspace, n_local, pid, nblocks | output: dummy
__aiv__ void _mlir_ciface_cannshmem_ag_push(memref_t<__gm__ half, 1> *a_local,
                                            memref_t<__gm__ half, 1> *workspace,
                                            int64_t n_local, int64_t pid,
                                            int64_t nblocks,
                                            memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  (void)nblocks;
  __gm__ half *src = a_local->aligned + a_local->offset;
  __gm__ half *ws = workspace->aligned + workspace->offset;
  const int me = aclshmem_my_pe();
  const int npes = aclshmem_n_pes();
  if (pid != 0 || n_local <= 0) {
    return;
  }

  const uint32_t nbytes =
      static_cast<uint32_t>(n_local * static_cast<int64_t>(sizeof(half)));
  // Symmetric dst address of *this* rank's slot; putmem delivers it to ``pe``.
  __gm__ void *my_slot =
      (__gm__ void *)(ws + static_cast<int64_t>(me) * n_local);

  aclshmemx_barrier_all_vec();

  for (int pe = 0; pe < npes; ++pe) {
    if (pe == me) {
      continue;
    }
    aclshmem_putmem(my_slot, (__gm__ void *)src, nbytes, pe);
  }

  aclshmemx_barrier_all_vec();
}

} // extern "C"
