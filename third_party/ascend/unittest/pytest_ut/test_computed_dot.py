"""Bitcast-derived vector values must reach Cube through valid GM/L1 buffers."""
import math

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def _rounded(x):
    bits = x.to(tl.int32, bitcast=True)
    return ((bits + 4095 + ((bits >> 13) & 1)) & -8192).to(tl.float32, bitcast=True)


@triton.jit
def _compensated(X, T, W, Y, Side, K: tl.constexpr, N: tl.constexpr, R: tl.constexpr, LEFT: tl.constexpr,
                 TWIDDLE: tl.constexpr, MATERIALIZE: tl.constexpr):
    batch = tl.program_id(1)
    r = tl.arange(0, K)[:, None]
    c = tl.arange(0, N)[None, :]
    offset = r * N + c
    xr = tl.load(X + batch * 2 * K * N + offset, r < R, 0.)
    xi = tl.load(X + batch * 2 * K * N + K * N + offset, r < R, 0.)
    if TWIDDLE:
        wr = tl.load(T + offset)
        wi = tl.load(T + K * N + offset)
        xr, xi = xr * wr - xi * wi, xr * wi + xi * wr
    hi_r = _rounded(xr)
    lo_r = _rounded(xr - hi_r)
    hi_i = _rounded(xi)
    lo_i = _rounded(xi - hi_i)
    # A live Vector consumer must keep its UB value, independent of L1 copies.
    tl.store(Side + batch * K * N + offset, hi_r)
    if MATERIALIZE:
        hi_r, lo_r = hi_r + 0., lo_r + 0.
        hi_i, lo_i = hi_i + 0., lo_i + 0.
    row = tl.arange(0, 2 * K)
    k = tl.arange(0, K)
    if LEFT:
        hr, lr = tl.trans(hi_r), tl.trans(lo_r)
        hi, li = tl.trans(hi_i), tl.trans(lo_i)
        a = tl.load(W + k[:, None] + row[None, :] * 2 * K)
        b = tl.load(W + 4 * K * K + k[:, None] + row[None, :] * 2 * K)
        y = tl.dot(hr, a, input_precision='hf32')
        y += tl.dot(lr, a, input_precision='hf32')
        y += tl.dot(hr, b, input_precision='hf32')
        a = tl.load(W + K + k[:, None] + row[None, :] * 2 * K)
        b = tl.load(W + 4 * K * K + K + k[:, None] + row[None, :] * 2 * K)
        v = tl.dot(hi, a, input_precision='hf32')
        v += tl.dot(li, a, input_precision='hf32')
        v += tl.dot(hi, b, input_precision='hf32')
        y += v
        yr = tl.trans(al.extract_slice(y, [0, 0], [N, K], [1, 1]))
        yi = tl.trans(al.extract_slice(y, [0, K], [N, K], [1, 1]))
    else:
        a = tl.load(W + row[:, None] * 2 * K + k[None, :])
        b = tl.load(W + 4 * K * K + row[:, None] * 2 * K + k[None, :])
        y = tl.dot(a, hi_r, input_precision='hf32')
        y += tl.dot(a, lo_r, input_precision='hf32')
        y += tl.dot(b, hi_r, input_precision='hf32')
        a = tl.load(W + row[:, None] * 2 * K + K + k[None, :])
        b = tl.load(W + 4 * K * K + row[:, None] * 2 * K + K + k[None, :])
        v = tl.dot(a, hi_i, input_precision='hf32')
        v += tl.dot(a, lo_i, input_precision='hf32')
        v += tl.dot(b, hi_i, input_precision='hf32')
        y += v
        yr = al.extract_slice(y, [0, 0], [K, N], [1, 1])
        yi = al.extract_slice(y, [K, 0], [K, N], [1, 1])
    tl.store(Y + batch * 2 * K * N + offset, yr, r < R)
    tl.store(Y + batch * 2 * K * N + K * N + offset, yi, r < R)


def _cpu_round(x):
    bits = x.contiguous().view(torch.int32)
    return ((bits + 4095 + ((bits >> 13) & 1)) & -8192).view(torch.float32)


@pytest.mark.parametrize('radix,n,twiddle,materialize', [(15, 256, True, False), (17, 128, False, False),
                                                         (8, 64, True, False), (15, 128, True, True)])
@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('auto', [False, True])
def test_computed_compensated_dot(radix, n, twiddle, materialize, left, auto):
    torch.manual_seed(20429)
    k, batch, guard = max(16, triton.next_power_of_2(radix)), 41, 32
    x = torch.randn(batch, 2, k, n)
    x[:, :, radix:] = 0
    phase = torch.randn(2, k, n)
    real, imag = x[:, 0], x[:, 1]
    if twiddle:
        real, imag = real * phase[0] - imag * phase[1], real * phase[1] + imag * phase[0]
    weight = torch.randn(2, radix, radix) / math.sqrt(radix)
    block = torch.zeros(2 * k, 2 * k)
    block[:radix, :radix] = weight[0]
    block[:radix, k:k + radix] = -weight[1]
    block[k:k + radix, :radix] = weight[1]
    block[k:k + radix, k:k + radix] = weight[0]
    high = _cpu_round(block)
    weights = torch.stack((high, _cpu_round(block - high))).npu()
    expected = torch.einsum('ok,bkn->bon', block.double(), torch.cat((real, imag), dim=1).double())
    size = x.numel()
    storage = torch.full((size + 2 * guard, ), -111., device='npu')
    output = storage[guard:-guard]
    side_storage = torch.full((batch * k * n + 2 * guard, ), -111., device='npu')
    side = side_storage[guard:-guard]
    source, tw = x.npu(), phase.npu()
    options = dict(num_warps=4, num_stages=1, multibuffer=False, enable_auto_blockify=auto, enable_fp_fusion=False)

    def launch():
        return _compensated[(1, batch)](source, tw, weights, output, side, k, n, radix, left, twiddle, materialize,
                                        **options)

    def verify(scale):
        torch.npu.synchronize()
        got = output.cpu().reshape(batch, 2, k, n)
        want = expected.reshape(batch, 2, k, n) * scale
        actual = got[:, :, :radix].double()
        reference = want[:, :, :radix]
        error = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference)
        assert error < 5e-7, error
        assert torch.all(got[:, :, radix:] == -111.)
        values = side.cpu().reshape(batch, k, n)
        # Recompute the scaled Vector result with the same FP32 arithmetic.
        raw = x * scale
        rr, ii = raw[:, 0], raw[:, 1]
        if twiddle:
            rr = rr * phase[0] - ii * phase[1]
        assert torch.equal(values.view(torch.int32), _cpu_round(rr).view(torch.int32))
        assert torch.all(storage[:guard].cpu() == -111.) and torch.all(storage[-guard:].cpu() == -111.)
        assert torch.all(side_storage[:guard].cpu() == -111.) and torch.all(side_storage[-guard:].cpu() == -111.)

    compiled = launch()
    verify(1.)
    assert str(compiled.asm['ttadapter']).count('memref_ext.alloc_workspace') == 4
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    for scale in (1e-6, 1e6, 1.):
        source.copy_((x * scale).npu())
        storage.fill_(-111.)
        side_storage.fill_(-111.)
        graph.replay()
        verify(scale)
    assert torch.equal(source.cpu(), x)


@triton.jit
def _both_inputs(A, B, C):
    r = tl.arange(0, 32)[:, None]
    k = tl.arange(0, 16)[None, :]
    a = _rounded(tl.load(A + r * 16 + k) * 1.125)
    b = _rounded(tl.load(B + tl.trans(k) * 32 + tl.trans(r)) * 0.875)
    value = tl.dot(a, b, input_precision='hf32')
    tl.store(C + r * 32 + tl.arange(0, 32)[None, :], value)


@pytest.mark.parametrize('auto', [False, True])
def test_both_dot_inputs_computed(auto):
    torch.manual_seed(1429)
    a, b = torch.randn(32, 16), torch.randn(16, 32)
    output = torch.empty((32, 32), device='npu')
    compiled = _both_inputs[(1, )](a.npu(), b.npu(), output, num_warps=4, num_stages=1, multibuffer=False,
                                   enable_auto_blockify=auto, enable_fp_fusion=False)
    torch.npu.synchronize()
    expected = _cpu_round(a * 1.125).double() @ _cpu_round(b * 0.875).double()
    assert torch.linalg.vector_norm(output.cpu().double() - expected) / torch.linalg.vector_norm(expected) < 5e-7
    assert str(compiled.asm['ttadapter']).count('memref_ext.alloc_workspace') == 2


@triton.jit
def _loaded_inputs(A, B, C):
    r = tl.arange(0, 32)[:, None]
    k = tl.arange(0, 16)[None, :]
    a = tl.load(A + r * 16 + k)
    b = tl.load(B + tl.trans(k) * 32 + tl.trans(r))
    tl.store(C + r * 32 + tl.arange(0, 32)[None, :], tl.dot(a, b, input_precision='hf32'))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('auto', [False, True])
def test_loaded_dot_keeps_native_path(dtype, auto):
    # Exactly HF32-representable inputs also provide a strict FP32 oracle.
    a = (torch.arange(512).remainder(31).float().reshape(32, 16) / 8).to(dtype)
    b = (torch.arange(512).remainder(23).float().reshape(16, 32) / 8).to(dtype)
    output = torch.empty((32, 32), device='npu')
    compiled = _loaded_inputs[(1, )](a.npu(), b.npu(), output, num_warps=4, num_stages=1, multibuffer=False,
                                     enable_auto_blockify=auto, enable_fp_fusion=False)
    torch.npu.synchronize()
    assert torch.equal(output.cpu().double(), a.double() @ b.double())
    assert 'memref_ext.alloc_workspace' not in str(compiled.asm['ttadapter'])


@triton.jit
def _batched_input(A, B, C):
    batch = tl.arange(0, 2)[:, None, None]
    r = tl.arange(0, 32)[None, :, None]
    k = tl.arange(0, 16)[None, None, :]
    a = _rounded(tl.load(A + batch * 512 + r * 16 + k) * 1.125)
    b = tl.load(B + batch * 512 + tl.arange(0, 16)[None, :, None] * 32 + tl.arange(0, 32)[None, None, :])
    value = tl.dot(a, b, input_precision='hf32')
    tl.store(C + batch * 1024 + r * 32 + tl.arange(0, 32)[None, None, :], value)


@pytest.mark.parametrize('auto', [False, True])
def test_batched_computed_dot(auto):
    torch.manual_seed(2429)
    a = torch.randn(2, 32, 16)
    b = _cpu_round(torch.randn(2, 16, 32))
    output = torch.empty((2, 32, 32), device='npu')
    compiled = _batched_input[(1, )](a.npu(), b.npu(), output, num_warps=4, num_stages=1, multibuffer=False,
                                     enable_auto_blockify=auto, enable_fp_fusion=False)
    torch.npu.synchronize()
    expected = _cpu_round(a * 1.125).double() @ b.double()
    assert torch.linalg.vector_norm(output.cpu().double() - expected) / torch.linalg.vector_norm(expected) < 5e-7
    assert str(compiled.asm['ttadapter']).count('memref_ext.alloc_workspace') == 1
