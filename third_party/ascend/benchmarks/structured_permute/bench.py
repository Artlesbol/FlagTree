"""One case/process. Exact checks precede timing; incorrect runs get no timing.

Baseline rows are contiguous float32, Q >= 32, a power of two. Physical strided
layouts are exercised separately by compiler regressions: tl.split's logical
axis must not be confused with HBM strides or UB strides chosen downstream.
"""
import argparse
import ctypes
import json
import os
import statistics
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def split_kernel(X, Z, Y, Q: tl.constexpr):
    r = tl.program_id(0)
    v = tl.load(X + r * 2 * Q + tl.arange(0, 2 * Q))
    a, b = tl.split(tl.reshape(v, (Q, 2)))
    i = tl.arange(0, Q)
    tl.store(Y + r * 2 * Q + i, a)
    tl.store(Y + r * 2 * Q + Q + i, b)


@triton.jit
def join_kernel(X, Z, Y, Q: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, Q)
    a = tl.load(X + r * Q + i)
    b = tl.load(Z + r * Q + i)
    v = tl.reshape(tl.join(a, b), (2 * Q,))
    tl.store(Y + r * 2 * Q + tl.arange(0, 2 * Q), v)


@triton.jit
def nested_kernel(X, Z, Y, T, J, Q: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, 4 * Q)
    v = tl.load(X + r * 4 * Q + i)
    t = tl.trans(tl.reshape(v, (4, Q)))
    tl.store(T + r * 4 * Q + i, tl.reshape(t, (4 * Q,)))
    lo, hi = tl.split(tl.reshape(t, (Q, 2, 2)))
    a, b = tl.split(lo)
    c, d = tl.split(hi)
    j = tl.arange(0, Q)
    tl.store(Y + r * 4 * Q + j, a)
    tl.store(Y + r * 4 * Q + Q + j, b)
    tl.store(Y + r * 4 * Q + 2 * Q + j, c)
    tl.store(Y + r * 4 * Q + 3 * Q + j, d)
    joined = tl.join(tl.join(a, b), tl.join(c, d))
    tl.store(J + r * 4 * Q + i, tl.reshape(joined, (4 * Q,)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["triton", "aiv"], required=True)
    p.add_argument("--case", choices=["split", "join", "nested"], required=True)
    p.add_argument("--q", type=int, default=256)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--library", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()
    if args.q < 32 or args.q > 4096 or args.q & (args.q - 1):
        p.error("Q must be a power of two in [32,4096]")
    if args.warmup < 3 or args.iters < 30 or args.rounds < 3:
        p.error("require warmup>=3, iters>=30, rounds>=3")
    torch.npu.set_device(0)
    torch.manual_seed(20261027)
    q, batch = args.q, args.batch
    width = (4 if args.case == "nested" else 2) * q
    xcpu = torch.randn(batch, width)
    zcpu = torch.randn(batch, q)
    x, z = xcpu.npu(), zcpu.npu()
    y = torch.full((batch, width), float("nan"), device="npu")
    t, j = torch.empty_like(y), torch.empty_like(y)
    if args.impl == "aiv":
        if not args.library or (args.case == "nested" and q > 1024):
            p.error("AIV reference needs --library; nested requires Q<=1024")
        lib = ctypes.CDLL(str(args.library.resolve()))
        call = lib.launch_permute
        call.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 3
        call.restype = ctypes.c_uint32
        idx = ((torch.arange(2*q) // 2) + (torch.arange(2*q) % 2) * q).to(torch.int32) * 4
        if args.case == "nested":
            i = torch.arange(4*q)
            idx = ((i // 4) + (i % 4) * q).to(torch.int32) * 4
        idx = idx.npu()
        stream = torch.npu.current_stream().npu_stream
        def launch():
            rc = call(min(batch, 40), stream, x.data_ptr(), z.data_ptr(), y.data_ptr(),
                      idx.data_ptr(), t.data_ptr(), j.data_ptr(), q, batch,
                      {"split": 0, "join": 1, "nested": 2}[args.case])
            if rc:
                raise RuntimeError(f"AIV launch failed: {rc}")
    elif args.case == "nested":
        def launch():
            return nested_kernel[(batch,)](x, z, y, t, j, Q=q)
    else:
        kernel = split_kernel if args.case == "split" else join_kernel
        def launch():
            return kernel[(batch,)](x, z, y, Q=q)
    launch()
    torch.npu.synchronize()
    if args.case == "split":
        expected = xcpu.reshape(batch, q, 2).transpose(1, 2).reshape(batch, 2*q)
    elif args.case == "join":
        # X is allocated with width 2Q; both implementations read the same
        # contiguous first batch*Q values, not an alternative row stride.
        a = xcpu.reshape(-1)[:batch*q].reshape(batch, q)
        expected = torch.stack((a, zcpu), dim=-1).reshape(batch, 2*q)
    else:
        expected = xcpu.reshape(batch, 4, q)[:, [0, 2, 1, 3], :].reshape(batch, 4*q)
        transposed = xcpu.reshape(batch, 4, q).transpose(1, 2).reshape(batch, 4*q)
        torch.testing.assert_close(t.cpu(), transposed, rtol=0, atol=0)
        torch.testing.assert_close(j.cpu(), transposed, rtol=0, atol=0)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()
    # Device graph makes the timing independent of Python vs ctypes launch cost.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    torch.npu.synchronize()
    times = []
    for _ in range(args.rounds):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / args.iters)
    result = vars(args).copy()
    result.update(status="pass", kernel_us=statistics.median(times), rounds_us=times,
                  us_per_row=statistics.median(times)/batch, triton=triton.__version__,
                  cache=os.environ.get("TRITON_CACHE_DIR"), timing="device_graph_events",
                  aiv_join="vector_gather_including_index_load")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
