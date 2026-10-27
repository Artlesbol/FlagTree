# Ascend split/join correctness and performance

This suite compares compiler-generated split/join with an independent Ascend C
vector implementation. It contains no FFT code or size-specific FFT rewrite.

## What is measured

| Case | Work per row | Observable outputs |
|---|---|---|
| `split` | Deinterleave `2Q` elements | Both channels |
| `join` | Interleave two independent `Q`-element inputs | Interleaved result |
| `nested` | Transpose `(4,Q)`, split twice, reconstruct | Transpose, four channels, reconstruction |

Both implementations read the same inputs and produce the same outputs. The AIV
reference uses `GatherMask` for split and vector `Gather` for join/transpose.
Its index-table HBM load is included. It is a vector reference, not a hardware
peak-throughput claim. `aiv.cpp` treats values as raw 16/32-bit words.

Each run checks every output before timing. After graph capture it clears all
outputs, replays the graph, and checks them again. Native launches resolve the
current stream inside the launch callback so they are captured on the graph's
stream. Each graph contains 64 kernel calls by default; event time is divided by the
actual call count. This amortizes host replay gaps. Use `--graph-steps` to check
convergence; all compared implementations must use the same protocol. JSON contains all timing rounds and replay verification.

## Build and run

Use a rebuilt Ascend FlagTree package and its matching patched FLIR dependency.
The CANN installation is not modified. Select a free NPU and run one case per
process: a runtime device exception can invalidate that process's NPU context.

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export FLAGTREE_BACKEND=ascend TRITON_BACKEND=torch_npu
export TRITON_JIT_BACKEND=NPU TRITON_ASCEND_ARCH=Ascend910B4
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1
export ASCEND_VISIBLE_DEVICES=6 ASCEND_RT_VISIBLE_DEVICES=6 NPU_VISIBLE_DEVICES=6
export TRITON_ALL_BLOCKS_PARALLEL=0
export TRITON_NPU_COMPILER_PATH=/usr/local/Ascend/cann-9.0.0/tools/bishengir/bin
# For an isolated in-tree build (adjust the platform/Python suffix):
export PYTHONPATH="$PWD/build/lib.linux-aarch64-cpython-311"
export TRITON_CACHE_DIR=/tmp/structured-permute-cache
bench=third_party/ascend/benchmarks/structured_permute
out=/tmp/structured-permute-results
mkdir -p "$out"
cmake -S "$bench" -B "$out/aiv-build" -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0 -DSOC_VERSION=Ascend910B4
cmake --build "$out/aiv-build" -j8

for impl in triton aiv; do
  for case in split join nested; do
    for q in 32 256 1024; do
      python "$bench/bench.py" --impl "$impl" --case "$case" --q "$q" \
        --batch 2048 --aiv-blocks 2048 --dtype float32 \
        --library "$out/aiv-build/libpermute_reference.so" \
        --output "$out/$impl-$case-q$q.json"
    done
  done
done
```

For a separate 40-block comparison, enable the existing compiler auto-blockify
option and compile a fresh kernel; use the persistent native reference alongside
it. Do not change this option while replaying an archived binary compiled with a
different dispatch contract.

```bash
TRITON_ALL_BLOCKS_PARALLEL=1 TRITON_ALWAYS_COMPILE=1 python "$bench/bench.py" \
  --impl triton --case nested --q 256 --output "$out/triton40-nested-q256.json"
python "$bench/bench.py" --impl aiv --case nested --q 256 --aiv-blocks 40 \
  --library "$out/aiv-build/libpermute_reference.so" \
  --output "$out/aiv40-nested-q256.json"
```

`--aiv-blocks 2048` matches the Triton logical grid. The default AIV grid is a
persistent 40-block loop; report that separately. Use `--dtype float16`,
`bfloat16`, `int16`, or `int32` for the other word types. `--dump-dir` saves the
Triton compiler stages before timing. Incorrect runs raise and produce no
successful timing JSON.

## Regression tests

From the FlagTree root, under the same environment:

```bash
pytest -q third_party/ascend/unittest/pytest_ut/test_split_join_layout.py -x
pytest -q third_party/ascend/unittest/pytest_ut/test_split.py \
  third_party/ascend/unittest/pytest_ut/test_interleave.py \
  third_party/ascend/unittest/pytest_ut/test_interleave_optimizaiton.py \
  third_party/ascend/unittest/pytest_ut/test_ir_builder.py -x
pytest -q third_party/ascend/unittest/pytest_ut/test_custom_bytecode_compat.py -x
pytest -q third_party/ascend/unittest/pytest_ut/test_split_join_values.py -x
```

The layout suite covers rank/shape changes, multiple consumers, independent and
swapped joins, input/output strides and offsets, output guard elements, computed
values, masked tails, constants, and exact floating-point bit patterns including
NaN payloads. It also exercises the native large-transpose fallback.

## Compiler implementation and limits

The FLIR pass first folds exact inverse `join(split(x))` pairs and moves channel
selection before transpose when the shape identity proves it safe. Small-axis
static transposes use vector index arithmetic and a private UB permutation
primitive; large transposes retain native tiling. These are static permutations,
not dynamic `tl.gather` operations.

Before dialect conversion, the pass proves single-use dense unmasked loads and
contiguous terminal stores. A simple rank-two split or rank-one word join meeting
that proof retains CANN's native load/store-folding path. Transposed, computed,
masked and shared inputs do not qualify. This is a graph property, not a
benchmark-name or size-specific admission rule.

In the packed and proven dense native paths, floating-point joins move values as same-width integers so concat/copy cannot
canonicalize NaN payloads. The c220 vector path is enabled for Ascend 910B targets;
other architectures and encoded join layouts retain the existing lowering.

Private device implementations are built into FlagTree's packaged
`custom_ops.bc` and linked automatically. Split uses `vreducev2` for packed data;
a gather honors the physical UB strides chosen downstream. An older CANN custom
operation schema is adapted at the bytecode boundary. After CANN has planned
buffers and synchronization, a private-operation-only ABI adapter emits device
function calls for the installed hivmc. It preserves existing allocations,
strides and synchronization; no installed CANN template or binary is patched.

The vector fast path covers supported 16/32-bit values with a packed destination,
including a strided source. Narrow byte values use private byte/word storage conversion around permutations,
which prevents CANN from moving a flat cast back into its broken sub-block 2D
cast path. Nonunit destination/index/scratch strides currently
have a synchronized scalar correctness path. Do not claim vector performance
for those layouts. Tiny two-scalar joins use two exact element insertions.

Static permutation indices are bounded to 4096 entries (16 KiB). Joins with
more than 2048 elements per input retain the existing interleave lowering before
any byte widening; larger transposes retain native tiling. Zero-offset rank-one
one-element copies normalize irrelevant strides to one, avoiding a separate
strided-byte-load failure without adding scalar data loops. This suite verifies the listed cases and does
not establish correctness or performance for every Triton operation/dtype or
other CANN/SoC versions.

Set `FLAGTREE_PERMUTE_DUMP_DIR` to retain the private operations immediately before
and after late ABI lowering. These artifacts expose actual UB descriptors and
synchronization, so the packed vector-path claim can be checked against the
physical strides rather than inferred from source tensor shapes.
