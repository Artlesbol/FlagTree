# Copyright 2026-     FlagOS Contributors
"""gemm-allreduce test wrapper.

Calls each impl's ``run_gemm_ar`` directly; checks/benches vs ``torch_npu``.

  GEMM_AR_IMPL=tle|raw|raw_hccl|raw_strip|raw_putmem|all

  ``tle`` and ``raw``/``raw_hccl`` share MM+HCCL (algo-aligned).
  raw_strip/putmem = raw-only candidates.

  torchrun --nproc_per_node=2|4|8 gemm_ar.py
  python gemm_ar.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
from triton.experimental.tle.raw.cann.cannshmem.utils import (
    cleanup_symmetric_tensor,
    create_symmetric_tensor,
    init_cannshmem_by_torch_pg,
)

import gemm_ar_raw
import gemm_ar_raw_hccl
import gemm_ar_raw_putmem
import gemm_ar_raw_strip
import gemm_ar_tle
from _common import (
    DEVICE,
    WARMUP,
    CaseResult,
    check_vs_npu,
    hccl_name,
    make_inputs,
    run_torch_npu,
    time_us,
)

DIR = Path(__file__).resolve().parent

# Production-like RowParallelLinear (LLM): N=hidden, K_per=hidden/TP, M=tokens.
# small M → comm-heavy (exposes oneshot vs AscendC tile-Commit gap);
# large M → prefill-ish compute.
CASES = {
    2: (
        ("small", 128, 4096, 2048),
        ("large", 2048, 4096, 2048),
    ),
    4: (
        ("small", 128, 4096, 1024),
        ("large", 2048, 4096, 1024),
    ),
    8: (
        ("small", 128, 4096, 512),
        ("large", 2048, 4096, 512),
    ),
}

_IMPL_MODS = {
    "tle": gemm_ar_tle,
    "raw": gemm_ar_raw,
    "raw_hccl": gemm_ar_raw_hccl,
    "raw_strip": gemm_ar_raw_strip,
    "raw_putmem": gemm_ar_raw_putmem,
}


def selected_backends():
    impl = os.environ.get("GEMM_AR_IMPL", "all").strip().lower()
    if impl in ("all", "*"):
        names = list(_IMPL_MODS)
    else:
        names = [x.strip() for x in impl.split(",") if x.strip()]
    unknown = [n for n in names if n not in _IMPL_MODS]
    if unknown:
        raise SystemExit(f"unknown GEMM_AR_IMPL {unknown}; choose from {list(_IMPL_MODS)} or all")
    return [(n, _IMPL_MODS[n]) for n in names]


def evaluate(
    mod,
    a,
    b,
    symm,
    out_flat,
    hcom: str,
    rank: int,
    world: int,
    tag: str,
    m: int,
    n: int,
    k: int,
) -> CaseResult:
    base = dict(backend=mod.BACKEND, world=world, tag=tag, m=m, n=n, k=k)
    if not getattr(mod, "IMPLEMENTED", False):
        return CaseResult(**base, status="SKIP", reason="not_implemented")

    out_mn = out_flat[:m * n].view(m, n)

    def _run():
        mod.run_gemm_ar(a, b, symm, out_flat, rank, world)

    dist.barrier()
    try:
        _run()
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001 — keep suite alive across impls
        if rank == 0:
            print(f"  {mod.BACKEND}: exception {type(exc).__name__}: {exc}", flush=True)
        return CaseResult(**base, status="FAIL", corr_ok=False, reason=str(exc)[:120])

    npu_out = run_torch_npu(a, b, hcom)
    torch.npu.synchronize()

    ok, max_abs, mean_abs = check_vs_npu(out_mn, npu_out)
    if not ok:
        return CaseResult(
            **base,
            status="FAIL",
            corr_ok=False,
            max_abs_vs_npu=max_abs,
            mean_abs_vs_npu=mean_abs,
        )

    for _ in range(WARMUP):
        _run()
        run_torch_npu(a, b, hcom)
    torch.npu.synchronize()
    dist.barrier()
    npu_us = time_us(lambda: run_torch_npu(a, b, hcom))
    impl_us = time_us(_run)
    return CaseResult(
        **base,
        status="PASS",
        corr_ok=True,
        max_abs_vs_npu=max_abs,
        mean_abs_vs_npu=mean_abs,
        impl_us=impl_us,
        npu_us=npu_us,
    )


def _print_table(results: list[CaseResult]) -> None:
    print(
        f"\n{'backend':<14} {'world':>5} {'tag':<7} {'MxNxK':<22} "
        f"{'corr':<6} {'impl_us':>10} {'npu_us':>10} {'speedup':>8} {'status':<6}",
        flush=True,
    )
    for r in results:
        shape = f"{r.m}x{r.n}x{r.k}"
        corr = {True: "PASS", False: "FAIL", None: "-"}.get(r.corr_ok, "-")
        impl = f"{r.impl_us:.1f}" if r.impl_us is not None else "-"
        npu = f"{r.npu_us:.1f}" if r.npu_us is not None else "-"
        sp = f"{r.speedup:.2f}x" if r.speedup is not None else "-"
        print(
            f"{r.backend:<14} {r.world:>5} {r.tag:<7} {shape:<22} "
            f"{corr:<6} {impl:>10} {npu:>10} {sp:>8} {r.status:<6}",
            flush=True,
        )


def run_worker() -> list[CaseResult]:
    init_cannshmem_by_torch_pg(ash_size=1024 * 1024 * 1024)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world not in CASES:
        raise SystemExit(f"no CASES for world={world}; supported {sorted(CASES)}")
    hcom = hccl_name(rank)

    max_m = max(m for _, m, _, _ in CASES[world])
    max_n = max(n for _, _, n, _ in CASES[world])
    max_elems = max_m * max_n
    backends = selected_backends()
    for _name, mod in backends:
        prep = getattr(mod, "prepare", None)
        if prep is not None:
            prep(rank, world, max_m, max_n)

    # One symmetric C buffer for the sweep (avoid per-shape finalize).
    symm = create_symmetric_tensor(torch.empty(max_elems, dtype=torch.float16, device=DEVICE))
    out_flat = torch.empty(max_elems, dtype=torch.float16, device=DEVICE)

    results: list[CaseResult] = []
    try:
        for tag, m, n, k in CASES[world]:
            if rank == 0:
                print(f"\n---- world={world} shape={tag}({m}x{n}x{k}) ----", flush=True)
            a, b = make_inputs(rank, m, n, k)
            for _name, mod in backends:
                r = evaluate(mod, a, b, symm, out_flat, hcom, rank, world, tag, m, n, k)
                results.append(r)
                if rank == 0:
                    print(
                        f"  {r.backend}: status={r.status} "
                        f"corr={r.corr_ok} max_abs={r.max_abs_vs_npu} "
                        f"impl_us={r.impl_us} npu_us={r.npu_us}",
                        flush=True,
                    )
    finally:
        cleanup_symmetric_tensor(symm)

    if rank == 0:
        _print_table(results)
        n_fail = sum(1 for r in results if r.status == "FAIL")
        n_pass = sum(1 for r in results if r.status == "PASS")
        print(f"PASS={n_pass} FAIL={n_fail} SKIP={len(results)-n_pass-n_fail}", flush=True)
    dist.barrier()
    # Only hard-fail if a primary impl (tle / raw / raw_hccl) failed.
    primary = {"tle", "raw", "raw_hccl"}
    if any(r.status == "FAIL" and r.backend in primary for r in results):
        raise SystemExit(1)
    return results


def _devices(world: int) -> str:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not visible:
        raise SystemExit("set ASCEND_RT_VISIBLE_DEVICES to ≥8 NPU ids, e.g. 0,1,2,3,4,5,6,7")
    ids = [x.strip() for x in visible.split(",") if x.strip()]
    if len(ids) < world:
        raise SystemExit(f"need {world} devices, got {len(ids)}: {visible!r}")
    return ",".join(ids[:world])


def run_driver() -> None:
    """Re-launch torchrun per world; worker prints tables to the same stdout."""
    torchrun = os.environ.get("TORCHRUN", "torchrun")
    port = 30200
    failed = False
    for world in CASES:
        port += 1
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = _devices(world)
        env["PYTHONPATH"] = f"{DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
        cmd = [
            torchrun,
            f"--nproc_per_node={world}",
            f"--master_port={port}",
            str(DIR / "gemm_ar.py"),
        ]
        print(
            f"\n==== world={world} devices={env['ASCEND_RT_VISIBLE_DEVICES']} ====",
            flush=True,
        )
        rc = subprocess.run(cmd, cwd=str(DIR), env=env).returncode
        if rc != 0:
            failed = True
    if failed:
        raise SystemExit(1)


def main():
    if "LOCAL_RANK" in os.environ:
        run_worker()
    else:
        run_driver()


if __name__ == "__main__":
    main()
