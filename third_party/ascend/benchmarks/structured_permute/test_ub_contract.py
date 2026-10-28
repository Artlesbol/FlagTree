"""Direct UB descriptor tests. Set FLAGTREE_UB_CONTRACT_LIBRARY explicitly.

These call the packaged implementation with controlled *physical* UB layouts.
They do not infer UB strides from HBM views. Guard checks cover whole buffers.
"""
import ctypes
import os
from pathlib import Path

import pytest
import torch
import torch_npu

STORAGE = 512
LAYOUTS = {
    'packed': (1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
    'strided_source': (3, 1, 1, 0, 1, 0, 1, 0, 1, 0),
    'strided_destinations': (1, 0, 2, 1, 3, 1, 1, 0, 1, 0),
    'strided_indices': (1, 0, 1, 0, 1, 0, 2, 1, 1, 0),
    'strided_scratch': (3, 1, 1, 0, 1, 0, 1, 0, 2, 1),
    'unaligned_source': (1, 1, 1, 0, 1, 0, 1, 0, 1, 0),
    'unaligned_destination': (1, 0, 1, 1, 1, 1, 1, 0, 1, 0),
    'unaligned_indices': (1, 0, 1, 0, 1, 0, 1, 1, 1, 0),
    'unaligned_scratch': (3, 1, 1, 0, 1, 0, 1, 0, 1, 1),
}


@pytest.fixture(scope='module')
def launch():
    torch.npu.set_device(0)
    path = Path(os.environ['FLAGTREE_UB_CONTRACT_LIBRARY']).resolve()
    assert path.is_file(), path
    library = ctypes.CDLL(str(path))
    call = library.launch_ub_contract
    call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 2
    call.restype = ctypes.c_uint32

    def run(x, indices, a, b, cfg, mode):
        rc = call(torch.npu.current_stream().npu_stream, x.data_ptr(), indices.data_ptr(), a.data_ptr(), b.data_ptr(),
                  cfg.data_ptr(), mode, x.element_size())
        assert rc == 0, rc
        torch.npu.synchronize()

    return run


@pytest.mark.parametrize('layout', list(LAYOUTS))
@pytest.mark.parametrize('dtype', [torch.int16, torch.int32])
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('n', [1, 7, 8, 9, 15, 16, 17, 64])
def test_word_descriptors(launch, layout, dtype, mode, n):
    ss, so, ds, do, bs, bo, ins, ino, ts, to = LAYOUTS[layout]
    cpu = (torch.arange(STORAGE, dtype=torch.int32) * 977 - 31001).to(dtype)
    indices = torch.zeros(STORAGE, dtype=torch.int32)
    logical_indices = (torch.arange(n) * 13 + 7).remainder(2 * n)
    indices[ino:ino + n * ins:ins] = logical_indices
    x, idx = cpu.npu(), indices.npu()
    a, b = [torch.full((STORAGE, ), -91, dtype=dtype, device='npu') for _ in range(2)]
    cfg = torch.tensor([n, 3, *LAYOUTS[layout]], dtype=torch.int32).npu()
    launch(x, idx, a, b, cfg, mode)
    ea, eb = [torch.full((STORAGE, ), -91, dtype=dtype) for _ in range(2)]
    if mode == 0:
        ea[do:do + n * ds:ds] = cpu[so + logical_indices * ss]
    else:
        ea[do:do + n * ds:ds] = cpu[so:so + 2 * n * ss:2 * ss]
        eb[bo:bo + n * bs:bs] = cpu[so + ss:so + (2 * n + 1) * ss:2 * ss]
    assert torch.equal(a.cpu(), ea), (mode, layout, n, a.cpu(), ea)
    assert torch.equal(b.cpu(), eb), (mode, layout, n, b.cpu(), eb)


@pytest.mark.parametrize('layout', [
    'packed', 'strided_source', 'strided_destinations', 'unaligned_source', 'unaligned_destination', 'unaligned_scratch'
])
@pytest.mark.parametrize('n', [1, 15, 16, 17, 64])
@pytest.mark.parametrize('mode', [2, 3])
def test_byte_storage_descriptors(launch, layout, n, mode):
    ss, so, ds, do, *_ = LAYOUTS[layout]
    src_dtype, dst_dtype = (torch.int8, torch.int16) if mode == 2 else (torch.int16, torch.int8)
    cpu = (torch.arange(STORAGE, dtype=torch.int32) * 37 + 211).to(torch.int8).to(src_dtype)
    x = cpu.npu()
    idx = torch.zeros(STORAGE, dtype=torch.int32, device='npu')
    a, b = [torch.full((STORAGE, ), -91, dtype=dst_dtype, device='npu') for _ in range(2)]
    cfg = torch.tensor([n, 3, *LAYOUTS[layout]], dtype=torch.int32).npu()
    launch(x, idx, a, b, cfg, mode)
    expected = torch.full((STORAGE, ), -91, dtype=dst_dtype)
    expected[do:do + n * ds:ds] = cpu[so:so + n * ss:ss].to(dst_dtype)
    assert torch.equal(a.cpu(), expected), (mode, layout, n)
