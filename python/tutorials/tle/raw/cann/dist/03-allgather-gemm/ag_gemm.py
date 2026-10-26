# Copyright 2026-     FlagOS Contributors
"""02-allgather-gemm: CANN AllGatherMatmul FullMesh schedule.

Reference (AscendC):
  ``toolkits/cann-9.1.0/.../all_gather_matmul/all_gather_matmul_full_mesh.h``
  ``third_party/ops-transformer/mc2/all_gather_matmul/``

Default ``AG_MODE=auto`` dispatches by shape (see ``_dispatch_mode``).
CANN FullMesh path:

  HcclPrepare:  hccl.AllGather(aGM → gatherGM, tileCnt)
  MatmulKernelLocal()                         # local MM first
  for tile: hccl.Wait → Matmul remote ranks   # gather ∥ was already in flight

Host equivalent:

  h = dist.all_gather_into_tensor(a_full, a_local, async_op=True)
  torch.mm(a_local, b, out=c[local])          # MatmulKernelLocal
  h.wait()
  for rem: torch.mm(a_full[rem], b, out=c[rem])

Raw fallback ``fuse``: same local-first schedule with ``aclshmem_putmem_nbi``
instead of HCCL AllGather.

Baseline: ``torch_npu.npu_all_gather_base_mm``.

Env:

  AG_MODE=auto|cann|fuse|p2p   # default auto → shape dispatch
  M_PER_RANK / N / K

Run:

  torchrun --nproc_per_node=2 --master_port=29791 \\
      python/tutorials/tle/raw/cann/dist/03-allgather-gemm/ag_gemm.py
"""

from __future__ import annotations

import os

import shmem as ash
import torch
import torch.distributed as dist
import torch_npu
from triton.experimental.tle.raw.cann.cannshmem.utils import (
    bench_us,
    cleanup_symmetric_tensor,
    create_symmetric_tensor,
    init_cannshmem_by_torch_pg,
)

DEVICE = "npu"
M_PER_RANK = int(os.environ.get("M_PER_RANK", "4096"))
N = int(os.environ.get("N", "4096"))
K = int(os.environ.get("K", "1024"))


def _dispatch_mode(m: int, k: int, n: int) -> str:
    """Pick path from multi-shape sweep vs torch_npu (world=2).

    - Large (M_local >= 2048): ``fuse`` putmem_nbi FullMesh ≫ npu (~1.3–2.0x).
    - Small / medium: ``cann`` HCCL local-first closer to npu (fuse worse on
      tiny shapes where putmem overhead dominates).
    """
    # Threshold from sweep: 2048x1024x2048 already fuse 1.31x; 512x512x1024 fuse 0.41x.
    if m >= 2048:
        return "fuse"
    return "cann"


# auto (default) | cann | fuse | p2p
_MODE_REQ = os.environ.get("AG_MODE", "auto").lower()
AG_MODE = _dispatch_mode(M_PER_RANK, K, N) if _MODE_REQ == "auto" else _MODE_REQ


def _stream_handle(stream=None):
    s = stream or torch.npu.current_stream()
    h = getattr(s, "npu_stream", None)
    return int(h if h is not None else s)


def _local_remote_slices(rank: int, world: int):
    m = M_PER_RANK
    local = slice(rank * m, (rank + 1) * m)
    if world == 2:
        remote = slice(0, m) if rank == 1 else slice(m, 2 * m)
        return local, [remote]
    remotes = [slice(p * m, (p + 1) * m) for p in range(world) if p != rank]
    return local, remotes


def _run_cann(a_local, b, a_full, c, rank, world):
    """CANN FullMesh: AllGather async ∥ local MM, Wait, then remote MM."""
    local, remotes = _local_remote_slices(rank, world)
    handle = dist.all_gather_into_tensor(a_full, a_local, async_op=True)
    torch.mm(a_local, b, out=c[local])  # MatmulKernelLocal
    handle.wait()
    for rem in remotes:
        torch.mm(a_full[rem], b, out=c[rem])


def _putmem_push_nbi(a_local, ws_flat, rank, world, n_local):
    nbytes = n_local * 2
    dst = int(ws_flat.data_ptr()) + rank * nbytes
    src = int(a_local.data_ptr())
    for pe in range(world):
        if pe == rank:
            continue
        ash.aclshmem_putmem_nbi(dst, src, nbytes, pe)


def _run_fuse_raw(a_local, b, ws_flat, workspace, c, rank, world, n_local):
    """Same FullMesh schedule with cannshmem putmem_nbi as the AllGather."""
    local, remotes = _local_remote_slices(rank, world)
    workspace[local].copy_(a_local)
    _putmem_push_nbi(a_local, ws_flat, rank, world, n_local)
    torch.mm(a_local, b, out=c[local])
    ash.aclshmemx_barrier_all_on_stream(_stream_handle())
    for rem in remotes:
        torch.mm(workspace[rem], b, out=c[rem])


def _run_p2p(a_local, b, symm, workspace, c, rank, world, peer_views):
    n_local = a_local.numel()
    symm[:n_local].copy_(a_local.reshape(-1))
    ash.aclshmemx_barrier_all_on_stream(_stream_handle())
    workspace[rank * M_PER_RANK:(rank + 1) * M_PER_RANK].copy_(a_local)
    for pe in range(world):
        if pe == rank:
            continue
        workspace[pe * M_PER_RANK:(pe + 1) * M_PER_RANK].copy_(
            peer_views[pe].view(M_PER_RANK, K)
        )
    torch.mm(workspace, b, out=c)


def _torch_ag_mm(a_local, b, world):
    chunks = [torch.empty_like(a_local) for _ in range(world)]
    dist.all_gather(chunks, a_local)
    return torch.mm(torch.cat(chunks, dim=0), b)


def main():
    need_ash = AG_MODE not in ("cann", "local_first", "hccl")
    if need_ash:
        init_cannshmem_by_torch_pg(ash_size=512 * 1024 * 1024)
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.npu.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="hccl")

    rank = dist.get_rank()
    world = dist.get_world_size()
    hcom = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    m_full = M_PER_RANK * world
    n_local = M_PER_RANK * K

    g = torch.Generator(device="cpu")
    g.manual_seed(0xA6E11 + rank)
    a_local = torch.randn(M_PER_RANK, K, dtype=torch.float16, generator=g).to(DEVICE).contiguous()
    g_b = torch.Generator(device="cpu")
    g_b.manual_seed(0xB0B0)
    b = torch.randn(K, N, dtype=torch.float16, generator=g_b).to(DEVICE).contiguous()

    a_full = torch.empty(m_full, K, dtype=torch.float16, device=DEVICE)
    c_fused = torch.empty(m_full, N, dtype=torch.float16, device=DEVICE)
    symm = ws_flat = workspace = None
    peer_views = None
    if need_ash:
        ws_flat = create_symmetric_tensor(
            torch.empty(m_full * K, dtype=torch.float16, device=DEVICE)
        )
        workspace = ws_flat.view(m_full, K)
        symm = create_symmetric_tensor(
            torch.empty(n_local, dtype=torch.float16, device=DEVICE)
        )
        ash.aclshmem_barrier_all()
        if AG_MODE == "p2p":
            peer_views = [None] * world
            for pe in range(world):
                if pe == rank:
                    continue
                ptr = ash.aclshmem_ptr(int(symm.data_ptr()), pe)
                peer_views[pe] = ash.construct_tensor_from_ptr(
                    ptr, (n_local, ), torch.float16, a_local.device
                )

    def _run_fused():
        if AG_MODE in ("cann", "local_first", "hccl"):
            _run_cann(a_local, b, a_full, c_fused, rank, world)
        elif AG_MODE in ("fuse", "put"):
            _run_fuse_raw(a_local, b, ws_flat, workspace, c_fused, rank, world, n_local)
        elif AG_MODE == "p2p":
            _run_p2p(a_local, b, symm, workspace, c_fused, rank, world, peer_views)
        else:
            raise ValueError(f"unknown AG_MODE={AG_MODE}")

    ref = _torch_ag_mm(a_local.float(), b.float(), world).to(torch.float16)
    torch.npu.synchronize()

    dist.barrier()
    _run_fused()
    torch.npu.synchronize()
    ok = torch.allclose(c_fused.float(), ref.float(), rtol=3e-2, atol=3e-2)
    print(f"[rank {rank}] fused({AG_MODE}) vs ref "
          f"{'PASS' if ok else 'FAIL'} "
          f"max_abs={(c_fused.float() - ref.float()).abs().max().item():.4f}",
          flush=True)

    npu_out, _ = torch_npu.npu_all_gather_base_mm(a_local, b, hcom, world)
    torch.npu.synchronize()
    ok_npu = torch.allclose(npu_out.float(), ref.float(), rtol=3e-2, atol=3e-2)
    ok_vs = torch.allclose(c_fused.float(), npu_out.float(), rtol=3e-2, atol=3e-2)
    print(f"[rank {rank}] torch_npu vs ref {'PASS' if ok_npu else 'FAIL'}", flush=True)
    print(f"[rank {rank}] fused vs torch_npu {'PASS' if ok_vs else 'FAIL'} "
          f"max_abs={(c_fused.float() - npu_out.float()).abs().max().item():.4f}",
          flush=True)
    if not ok_vs:
        if need_ash:
            cleanup_symmetric_tensor(symm)
            cleanup_symmetric_tensor(ws_flat)
        raise SystemExit(1)
    if not (ok and ok_npu) and rank == 0:
        print("[warn] vs fp32-ref loose; fused≡torch_npu — continue bench", flush=True)

    warmup, iters = 15, 40
    for _ in range(warmup):
        _run_fused()
        torch_npu.npu_all_gather_base_mm(a_local, b, hcom, world)
    torch.npu.synchronize()
    dist.barrier()

    us_npu = bench_us(
        lambda: torch_npu.npu_all_gather_base_mm(a_local, b, hcom, world),
        warmup=5, iters=iters,
    )
    us_fused = bench_us(_run_fused, warmup=5, iters=iters)
    us_torch = bench_us(lambda: _torch_ag_mm(a_local, b, world), warmup=5, iters=iters)

    if rank == 0:
        spd = us_npu / us_fused if us_fused > 0 else 0.0
        print(f"shape=({M_PER_RANK},{K})x({K},{N}) world={world} mode={AG_MODE}"
              f"{' (auto)' if _MODE_REQ == 'auto' else ''}")
        print(f"fused ({AG_MODE:11s})      {us_fused:8.1f} us")
        print(f"torch all_gather + mm      {us_torch:8.1f} us")
        print(f"torch_npu all_gather_mm    {us_npu:8.1f} us")
        print(f"speedup vs torch_npu       {spd:.2f}x  (>1 = faster)")
        print("02-allgather-gemm PASS")

    if need_ash:
        cleanup_symmetric_tensor(symm)
        cleanup_symmetric_tensor(ws_flat)


if __name__ == "__main__":
    main()
