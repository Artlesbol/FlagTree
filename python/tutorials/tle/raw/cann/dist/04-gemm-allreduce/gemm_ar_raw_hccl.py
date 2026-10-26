# Copyright 2026-     FlagOS Contributors
"""Raw entry for Cube MM + HCCL AllReduce (shared with ``tle``)."""

from __future__ import annotations

from _algo_mm_hccl import run_mm_hccl_ar

BACKEND = "raw_hccl"
IMPLEMENTED = True


def prepare(*_a, **_k):
    return


def run_gemm_ar(a, b, symm, out_flat, rank: int, world: int) -> None:
    del symm, rank, world
    run_mm_hccl_ar(a, b, out_flat)
