# Copyright 2026-     FlagOS Contributors
"""Strip Cube MM + raw mark + raw oneshot reduce (single stream, correct first).

Overlap across strips can be re-enabled once signal path is stable.
"""

from __future__ import annotations

import torch
from triton.experimental.tle.raw.cann.cannshmem.utils import create_symmetric_tensor

from _common import DEVICE
from _strip_util import launch_ar_strip, launch_mark, strip_plan

BACKEND = "raw_strip"
# Signal oneshot times out on large strips (AIV hang); keep for small-only experiments.
IMPLEMENTED = False

_ready = None
_epoch = 0


def prepare(rank: int, world: int, max_m: int, max_n: int) -> None:
    del rank, max_n
    global _ready
    nstrips = max(len(strip_plan(max_m)), 8)
    if _ready is None:
        _ready = create_symmetric_tensor(
            torch.zeros(nstrips * world, dtype=torch.int32, device=DEVICE)
        )


def run_gemm_ar(a, b, symm, out_flat, rank: int, world: int) -> None:
    global _epoch
    del rank
    m, n = a.shape[0], b.shape[1]
    strips = strip_plan(m)
    _epoch += 1
    epoch = _epoch
    for i, (row0, nrows) in enumerate(strips):
        torch.mm(
            a[row0:row0 + nrows], b,
            out=symm[row0 * n:(row0 + nrows) * n].view(nrows, n),
        )
        launch_mark(_ready, i, epoch, world)
        launch_ar_strip(
            symm, out_flat, _ready, i, epoch, world,
            row0 * n, nrows * n,
        )
