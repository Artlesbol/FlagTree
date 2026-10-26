# Copyright 2026-     FlagOS Contributors
"""Cube MM + chunked ``aclshmem_putmem_nbi`` FullMesh + Triton reduce."""

from __future__ import annotations

import shmem as ash
import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw.cann.cannshmem.utils import create_symmetric_tensor

from _common import AR_BLOCK, DEVICE, NUM_AR_CORES

BACKEND = "raw_putmem"
# Large puts still corrupt / incomplete across ranks; disable until quiet API wired.
IMPLEMENTED = False

_gather = None
_max_elems = 0
# cannshmem putmem often happier with modest chunks
_PUT_CHUNK = 256 * 1024  # half elems (~512KB)


def _kw():
    return dict(enable_legacy_insert_load_store_for_mix_cv=True)


def _stream_handle(stream=None):
    s = stream or torch.npu.current_stream()
    h = getattr(s, "npu_stream", None)
    return int(h if h is not None else s)


@triton.jit
def gather_reduce_kernel(
    gather_ptr, out_ptr, n_elems,
    WORLD: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)
    for start in range(pid * BLOCK, n_elems, ncore * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_elems
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for pe in tl.static_range(0, WORLD):
            acc += tl.load(
                gather_ptr + pe * STRIDE + offs, mask=mask, other=0.0
            ).to(tl.float32)
        tl.store(out_ptr + offs, acc.to(tl.float16), mask=mask)


def prepare(rank: int, world: int, max_m: int, max_n: int) -> None:
    del rank
    global _gather, _max_elems
    need = max_m * max_n
    if _gather is None:
        _max_elems = need
        _gather = create_symmetric_tensor(
            torch.empty(world * need, dtype=torch.float16, device=DEVICE)
        )


def run_gemm_ar(a, b, symm, out_flat, rank: int, world: int) -> None:
    del symm
    m, n = a.shape[0], b.shape[1]
    n_elems = m * n
    local = torch.empty(m, n, dtype=torch.float16, device=DEVICE)
    torch.mm(a, b, out=local)
    _gather.view(world, _max_elems)[rank, :n_elems].copy_(local.view(-1))
    base = int(_gather.data_ptr()) + rank * _max_elems * 2
    src0 = int(local.data_ptr())
    for off in range(0, n_elems, _PUT_CHUNK):
        chunk = min(_PUT_CHUNK, n_elems - off)
        dst = base + off * 2
        src = src0 + off * 2
        nbytes = chunk * 2
        for pe in range(world):
            if pe == rank:
                continue
            ash.aclshmem_putmem_nbi(dst, src, nbytes, pe)
    ash.aclshmemx_barrier_all_on_stream(_stream_handle())
    gather_reduce_kernel[(NUM_AR_CORES,)](
        _gather, out_flat, n_elems,
        WORLD=world, STRIDE=_max_elems, BLOCK=AR_BLOCK, **_kw(),
    )
