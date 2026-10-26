# Copyright 2026-     FlagOS Contributors
"""TLE entry — same algorithm as large-shape raw (Cube MM + HCCL AllReduce).

Aligned with ``raw_hccl`` on purpose: suite diffs then show raw-only wins
(signal / putmem / …), not algo mismatch.
"""

from __future__ import annotations

from _algo_mm_hccl import run_mm_hccl_ar

BACKEND = "tle"
IMPLEMENTED = True


def prepare(*_a, **_k):
    return


def run_gemm_ar(a, b, symm, out_flat, rank: int, world: int) -> None:
    del symm, rank, world
    run_mm_hccl_ar(a, b, out_flat)
