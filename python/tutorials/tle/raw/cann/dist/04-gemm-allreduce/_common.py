# Copyright 2026-     FlagOS Contributors
"""Shared helpers for gemm-ar internal tests (baseline / corr / bench)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_npu
from triton.experimental.tle.raw.cann.cannshmem.utils import bench_us

DEVICE = "npu"
NUM_AR_CORES = 16
AR_BLOCK = 2048
WARMUP = 10
ITERS = 30
MEAN_VS_NPU_MAX = 0.05
MAX_VS_NPU_MAX = 1.0


@dataclass
class CaseResult:
    backend: str
    world: int
    tag: str
    m: int
    n: int
    k: int
    status: str  # PASS | FAIL | SKIP
    corr_ok: bool | None = None
    max_abs_vs_npu: float | None = None
    mean_abs_vs_npu: float | None = None
    impl_us: float | None = None
    npu_us: float | None = None
    reason: str | None = None

    @property
    def speedup(self) -> float | None:
        if self.impl_us and self.npu_us and self.impl_us > 0:
            return self.npu_us / self.impl_us
        return None


def make_inputs(rank: int, m: int, n: int, k: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(0xC0FFEE + rank)
    a = torch.randn(m, k, dtype=torch.float16, generator=g).to(DEVICE).contiguous()
    b = torch.randn(k, n, dtype=torch.float16, generator=g).to(DEVICE).contiguous()
    return a, b


def hccl_name(rank: int) -> str:
    return dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank)


def run_torch_npu(a, b, hcom: str):
    return torch_npu.npu_mm_all_reduce_base(a, b, hcom, reduce_op="sum")


def check_vs_npu(out_mn, npu_out) -> tuple[bool, float, float]:
    d = (out_mn.float() - npu_out.float()).abs()
    max_abs = d.max().item()
    mean_abs = d.mean().item()
    ok = mean_abs < MEAN_VS_NPU_MAX and max_abs < MAX_VS_NPU_MAX
    return ok, max_abs, mean_abs


def time_us(fn) -> float:
    return bench_us(fn, warmup=5, iters=ITERS)
