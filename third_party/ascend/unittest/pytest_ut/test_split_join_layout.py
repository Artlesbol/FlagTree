"""Split/join layout regressions, including users that prevent load folding."""
import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _layout(X, Z, Y, AOUT, BOUT, TOUT, JOUT,
            A: tl.constexpr, B: tl.constexpr, XS: tl.constexpr, YS: tl.constexpr,
            SWAP: tl.constexpr):
    i = tl.arange(0, 2 * A * B)
    x = tl.load(X + i * XS)
    # Select a non-last axis in the original three dimensional tensor.
    t = tl.permute(tl.reshape(x, (A, 2, B)), (0, 2, 1))
    tl.store(TOUT + i, tl.reshape(t, (2 * A * B,)))
    a, b = tl.split(t)
    j = tl.arange(0, A * B)
    tl.store(AOUT + j, tl.reshape(a, (A * B,)))
    tl.store(BOUT + j, tl.reshape(b, (A * B,)))
    joined = tl.join(b, a) if SWAP else tl.join(a, b)
    tl.store(Y + i * YS, tl.reshape(joined, (2 * A * B,)))
    # An independent join, not eligible for join(split(x)) cancellation.
    z = tl.load(Z + tl.arange(0, A * B))
    other = tl.join(a, tl.reshape(z, (A, B)))
    tl.store(JOUT + i, tl.reshape(other, (2 * A * B,)))


@pytest.mark.parametrize("shape", [(1, 1), (2, 4), (4, 2), (4, 8), (8, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32,
                                    torch.int16, torch.int32, torch.int8])
@pytest.mark.parametrize("swap,strides", [(False, (1, 1)), (True, (3, 2))])
def test_split_join_layout(shape, dtype, swap, strides):
    a, b = shape
    xs, ys = strides
    n = 2 * a * b
    torch.manual_seed(123)
    # Random integers are exactly representable in all selected dtypes.
    storage = torch.randint(-100, 100, (n * xs + 2,), dtype=torch.int32).to(dtype).npu()
    x = storage[1:1+n*xs:xs]
    zcpu = torch.randint(-100, 100, (a*b,), dtype=torch.int32).to(dtype)
    z = zcpu.npu()
    out_storage = torch.full((n*ys + 2,), -111, dtype=dtype, device="npu")
    y = out_storage[1:1+n*ys:ys]
    ao = torch.empty(a*b, dtype=dtype, device="npu")
    bo = torch.empty_like(ao)
    to = torch.empty(n, dtype=dtype, device="npu")
    jo = torch.empty_like(to)
    _layout[(1,)](x, z, y, ao, bo, to, jo, A=a, B=b, XS=xs, YS=ys, SWAP=swap)
    cpu = x.cpu().reshape(a, 2, b)
    expected_a, expected_b = cpu[:, 0, :], cpu[:, 1, :]
    t = cpu.permute(0, 2, 1).contiguous().reshape(-1)
    joined = torch.stack((expected_b, expected_a) if swap else
                         (expected_a, expected_b), dim=-1).reshape(-1)
    other = torch.stack((expected_a, zcpu.reshape(a, b)), dim=-1).reshape(-1)
    for name, actual, expected in [("split0", ao, expected_a.reshape(-1)),
                                  ("split1", bo, expected_b.reshape(-1)),
                                  ("transpose", to, t), ("join", y, joined),
                                  ("independent_join", jo, other)]:
        assert torch.equal(actual.cpu(), expected), (name, actual.cpu(), expected)
    # Check all output gaps as well as both guard elements.
    guard = torch.ones(n*ys + 2, dtype=torch.bool)
    guard[1:1+n*ys:ys] = False
    assert torch.all(out_storage.cpu()[guard] == -111)
