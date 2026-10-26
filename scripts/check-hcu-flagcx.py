#!/usr/bin/env python3
"""Check the HCU host library and compile/run a call linked from FlagCX BC."""

import ctypes
import os

import torch
import triton
import triton.language as tl
from triton._flagtree_backend import FLAGTREE_BACKEND
from triton.runtime import driver


@triton.jit
def check_device_library(output, BC: tl.constexpr):
    # flagcxChar == 0; its size is one byte. This function is independent of
    # communicators, so linking can be checked before setting up multiple ranks.
    size = tl.extern_elementwise(
        lib_name="flagcx",
        lib_path=BC,
        args=[tl.full((), 0, tl.int32)],
        arg_type_symbol_dict={(tl.int32, ): ("flagcxDataTypeSizeDevice", tl.uint64)},
        is_pure=True,
    )
    tl.store(output, size.to(tl.int64))


def main():
    if FLAGTREE_BACKEND != "hcu" or not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Run this check with FlagTree HCU and HIP PyTorch on a visible Hygon GPU.")
    from triton.backends.hcu.distributed import Distributed, flagcx_rt_conf

    target = driver.active.get_current_target()
    if not flagcx_rt_conf.is_available:
        raise RuntimeError("The HCU FlagCX bundle is missing. Rebuild FlagTree with USE_FLAGCX=ON.")
    extern_libs = Distributed().get_extern_libs(target.arch)
    host = ctypes.CDLL(str(flagcx_rt_conf.shared_lib_path), mode=os.RTLD_NOW)
    host.flagcxGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    host.flagcxGetVersion.restype = ctypes.c_int
    version = ctypes.c_int()
    result = host.flagcxGetVersion(ctypes.byref(version))
    if result != 0:
        raise RuntimeError(f"flagcxGetVersion failed: {result}")
    print(f"PASS host: {flagcx_rt_conf.shared_lib_path}, version={version.value}")

    output = torch.empty(1, dtype=torch.int64, device="cuda")
    # Force a compile to exercise the linker even if this check was run before.
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.always_compile = True
        check_device_library[(1, )](output, BC=extern_libs["flagcx"], num_warps=1)
    torch.cuda.synchronize()
    if output.item() != 1:
        raise RuntimeError(f"FlagCX device-library call returned {output.item()}, expected 1")
    print(f"PASS device: linked and executed FlagCX bitcode on {target.arch}")
    print("Multi-rank communication and synchronization still require separate tests.")


if __name__ == "__main__":
    main()
