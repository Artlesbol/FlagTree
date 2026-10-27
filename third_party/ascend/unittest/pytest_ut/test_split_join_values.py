"""Value provenance and bit-preservation checks for structured permutations."""
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

from test_split_join_layout import _layout


@triton.jit
def _computed_nested(X, Y, T, J, Q: tl.constexpr):
    i = tl.arange(0, 4 * Q)
    v = (tl.load(X + i) + 1).to(X.dtype.element_ty)
    t = tl.trans(tl.reshape(v, (4, Q)))
    tl.store(T + i, tl.reshape(t, (4 * Q,)))
    lo, hi = tl.split(tl.reshape(t, (Q, 2, 2)))
    a, b = tl.split(lo)
    c, d = tl.split(hi)
    j = tl.arange(0, Q)
    tl.store(Y + j, a)
    tl.store(Y + Q + j, b)
    tl.store(Y + 2 * Q + j, c)
    tl.store(Y + 3 * Q + j, d)
    joined = tl.join(tl.join(a, b), tl.join(c, d))
    tl.store(J + i, tl.reshape(joined, (4 * Q,)))


@pytest.mark.parametrize('q', [4, 32, 256])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32,
                                   torch.int8, torch.int16, torch.int32])
def test_computed_transpose_nested_split(q, dtype):
    cpu = (torch.arange(4*q, dtype=torch.int32).remainder(97)-48).to(dtype)
    x = cpu.npu()
    y, t, j = [torch.empty_like(x) for _ in range(3)]
    _computed_nested[(1,)](x, y, t, j, Q=q)
    ref = (cpu.to(torch.int32)+1).to(dtype).reshape(4, q)
    assert torch.equal(y.cpu(), ref[[0, 2, 1, 3]].reshape(-1))
    transposed = ref.T.contiguous().reshape(-1)
    assert torch.equal(t.cpu(), transposed)
    assert torch.equal(j.cpu(), transposed)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('xs,ys', [(1, 1), (3, 2)])
def test_split_join_preserves_float_bits(dtype, xs, ys):
    bits_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    patterns = ([0, -2147483648, 0x7fc00001, -8388608, 0x7f800000, 1, -1, 12345]
                if dtype == torch.float32 else
                [0, -32768, 0x7fc1, -128, 0x7f80, 1, -1, 12345]
                if dtype == torch.bfloat16 else
                [0, -32768, 0x7e01, -1024, 0x7c00, 1, -1, 12345])
    raw = torch.tensor(patterns, dtype=bits_dtype).repeat(16)
    n, a, b = raw.numel(), 4, 16
    storage = torch.zeros(n*xs+2, dtype=bits_dtype)
    storage[1:1+n*xs:xs] = raw
    x = storage.view(dtype).npu()[1:1+n*xs:xs]
    zraw = raw[:n//2].flip(0).contiguous()
    z = zraw.view(dtype).npu()
    out = torch.zeros(n*ys+2, dtype=bits_dtype).view(dtype).npu()
    y = out[1:1+n*ys:ys]
    ao, bo = [torch.empty(n//2, dtype=dtype, device='npu') for _ in range(2)]
    to, jo = [torch.empty(n, dtype=dtype, device='npu') for _ in range(2)]
    _layout[(1,)](x, z, y, ao, bo, to, jo, A=a, B=b, XS=xs, YS=ys, SWAP=True)
    ref = raw.reshape(a, 2, b)
    ra, rb = ref[:, 0, :], ref[:, 1, :]
    expected = [torch.stack((rb, ra), dim=-1).reshape(-1), ra.reshape(-1), rb.reshape(-1),
                ref.permute(0, 2, 1).contiguous().reshape(-1),
                torch.stack((ra, zraw.reshape(a, b)), dim=-1).reshape(-1)]
    for value, bits in zip((y, ao, bo, to, jo), expected):
        assert torch.equal(value.cpu().view(bits_dtype), bits)
    guards = torch.ones(n*ys+2, dtype=torch.bool)
    guards[1:1+n*ys:ys] = False
    assert torch.all(out.cpu().view(bits_dtype)[guards] == 0)


@pytest.mark.parametrize('q', [1, 8, 64])
@pytest.mark.parametrize('xs,ys', [(1, 1), (3, 2)])
def test_unsigned_byte_channels(q, xs, ys):
    n = 4*q
    raw = (torch.arange(n*xs+2, dtype=torch.int32)*37+211).remainder(256).to(torch.uint8)
    x = raw.npu()[1:1+n*xs:xs]
    zcpu = (torch.arange(n//2, dtype=torch.int32)*19+177).remainder(256).to(torch.uint8)
    z = zcpu.npu()
    storage = torch.full((n*ys+2,), 111, dtype=torch.uint8, device='npu')
    y = storage[1:1+n*ys:ys]
    ao, bo = [torch.empty(n//2, dtype=torch.uint8, device='npu') for _ in range(2)]
    to, jo = [torch.empty(n, dtype=torch.uint8, device='npu') for _ in range(2)]
    _layout[(1,)](x, z, y, ao, bo, to, jo, A=2, B=q, XS=xs, YS=ys, SWAP=True)
    ref = x.cpu().reshape(2, 2, q)
    ra, rb = ref[:, 0, :], ref[:, 1, :]
    expected = [torch.stack((rb, ra), dim=-1).reshape(-1), ra.reshape(-1), rb.reshape(-1),
                ref.permute(0, 2, 1).contiguous().reshape(-1),
                torch.stack((ra, zcpu.reshape(2, q)), dim=-1).reshape(-1)]
    for value, reference in zip((y, ao, bo, to, jo), expected):
        assert torch.equal(value.cpu(), reference)
    guards = torch.ones(n*ys+2, dtype=torch.bool)
    guards[1:1+n*ys:ys] = False
    assert torch.all(storage.cpu()[guards] == 111)
