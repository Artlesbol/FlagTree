# Copyright 2026-     FlagOS Contributors
"""Shared algorithm: Cube MM + HCCL AllReduce (large-shape raw path).

Used by both ``tle`` and ``raw_hccl`` so suite can isolate algorithm vs
raw-only primitives (signal / putmem / …).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from _strip_util import strip_plan

_S_MM = None
_S_AR = None
_EVENTS = None


def _streams(n: int):
    global _S_MM, _S_AR, _EVENTS
    if _S_MM is None:
        _S_MM = torch.npu.Stream()
        _S_AR = torch.npu.Stream()
    if _EVENTS is None or len(_EVENTS) < n:
        _EVENTS = [torch.npu.Event() for _ in range(n)]
    return _S_MM, _S_AR, _EVENTS


def _nstrips(m: int) -> int:
    env = os.environ.get("RAW_HCCL_STRIPS")
    if env is not None:
        return max(int(env), 1)
    del m
    return 1


def run_mm_hccl_ar(a, b, out_flat) -> None:
    m, n = a.shape[0], b.shape[1]
    out = out_flat[:m * n].view(m, n)
    strips = strip_plan(m, _nstrips(m))

    if len(strips) == 1:
        torch.mm(a, b, out=out)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return

    s_mm, s_ar, events = _streams(len(strips))
    cur = torch.npu.current_stream()
    s_mm.wait_stream(cur)
    s_ar.wait_stream(cur)
    handles = []
    with torch.npu.stream(s_mm):
        for i, (row0, nrows) in enumerate(strips):
            torch.mm(a[row0:row0 + nrows], b, out=out[row0:row0 + nrows])
            events[i].record(s_mm)
    with torch.npu.stream(s_ar):
        for i, (row0, nrows) in enumerate(strips):
            s_ar.wait_event(events[i])
            handles.append(dist.all_reduce(out[row0:row0 + nrows], op=dist.ReduceOp.SUM, async_op=True))
        for h in handles:
            h.wait()
    cur.wait_stream(s_mm)
    cur.wait_stream(s_ar)
