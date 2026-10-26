# Copyright 2026-     FlagOS Contributors
"""Shared strip planning + raw mark/wait/ar_strip dialect wrappers."""

from __future__ import annotations

from pathlib import Path

import triton
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
from triton.experimental.tle.raw import dialect

_CPP = Path(__file__).parent / "ar_strip.cpp"
NUM_AR_CORES = 16


def strip_plan(m: int, num_strips: int | None = None) -> list[tuple[int, int]]:
    if num_strips is None:
        if m <= 128:
            num_strips = 4
        elif m <= 512:
            num_strips = 4
        else:
            num_strips = 8
    n = max(int(num_strips), 1)
    base, rem = divmod(m, n)
    out, row = [], 0
    for i in range(n):
        rows = base + (1 if i < rem else 0)
        if rows > 0:
            out.append((row, rows))
        row += rows
    return out


@dialect(
    name="cann",
    library="cannshmem",
    file=_CPP,
    extern_func_name="cannshmem_mark_strip",
    pipeline={"core": "vector", "pipe": "PIPE_MTE2"},
)
def _mark_strip_op(*args, **kwargs):
    ...


@dialect(
    name="cann",
    library="cannshmem",
    file=_CPP,
    extern_func_name="cannshmem_wait_strip",
    pipeline={"core": "vector", "pipe": "PIPE_MTE2"},
)
def _wait_strip_op(*args, **kwargs):
    ...


@dialect(
    name="cann",
    library="cannshmem",
    file=_CPP,
    extern_func_name="cannshmem_ar_strip",
    pipeline={"core": "vector", "pipe": "PIPE_MTE2"},
)
def _ar_strip_op(*args, **kwargs):
    ...


def _kw():
    return dict(enable_legacy_insert_load_store_for_mix_cv=True)


@triton.jit
def mark_strip_kernel(ready, strip, epoch, world):
    pid = tl.program_id(0)
    dummy = tl.zeros([16], dtype=tl.float16)
    _ = tle_raw.call(
        _mark_strip_op,
        [ready, strip, epoch, world, pid, dummy],
        output_indices=[5],
    )


@triton.jit
def wait_strip_kernel(ready, strip, epoch, world):
    pid = tl.program_id(0)
    dummy = tl.zeros([16], dtype=tl.float16)
    _ = tle_raw.call(
        _wait_strip_op,
        [ready, strip, epoch, world, pid, dummy],
        output_indices=[5],
    )


@triton.jit
def ar_strip_kernel(
    symm, out, ready, strip, epoch, world, elem_off, elem_n,
    NUM_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    dummy = tl.zeros([16], dtype=tl.float16)
    _ = tle_raw.call(
        _ar_strip_op,
        [symm, out, ready, strip, epoch, world, elem_off, elem_n, pid, NUM_BLOCKS, dummy],
        output_indices=[10],
    )


def launch_mark(ready, strip, epoch, world):
    mark_strip_kernel[(1,)](ready, strip, epoch, world, **_kw())


def launch_wait(ready, strip, epoch, world):
    wait_strip_kernel[(NUM_AR_CORES,)](ready, strip, epoch, world, **_kw())


def launch_ar_strip(symm, out, ready, strip, epoch, world, elem_off, elem_n):
    ar_strip_kernel[(NUM_AR_CORES,)](
        symm, out, ready, strip, epoch, world, elem_off, elem_n,
        NUM_BLOCKS=NUM_AR_CORES, **_kw(),
    )
