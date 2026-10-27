"""CPU checks for the current-MLIR / installed-CANN transport boundary."""
from types import SimpleNamespace
import importlib

import pytest
import torch_npu  # registers torch.npu for the suite's device fixture

compiler = importlib.import_module("triton.backends.ascend.compiler")
late = importlib.import_module("triton.backends.ascend.structured_permute_ir")


def test_nested_attribute_split():
    fields = compiler._split_mlir_attributes(
        'name = "comma, quote\\\"", mapping = affine_map<(d0) -> (d0)>, '
        'args = [{a = 1 : i32}, {b = "x,y"}], flag')
    assert len(fields) == 4
    assert fields[-1] == "flag"


def test_restore_legacy_segments():
    text = ('%r:2 = "cann_compat.custom"(%a, %b, %c, %d) '
            '{cann_compat.properties = {name = "test", no_side_effect, '
            'operandSegmentSizes = array<i32: 2, 2, 0>}, symbol = "impl"} '
            ': (tensor<8xf32>, tensor<8xi32>, tensor<8xf32>, tensor<8xi32>) '
            '-> (tensor<8xf32>, tensor<8xi32>)\n')
    restored = compiler._restore_cann_custom_ops(text, 2)
    assert '"hivm.hir.custom"' in restored
    assert "operandSegmentSizes = array<i32: 2, 2>" in restored
    assert '<{name = "test", no_side_effect,' in restored
    assert 'symbol = "impl"' in restored
    with pytest.raises(RuntimeError, match="temp-buffer"):
        compiler._restore_cann_custom_ops(text.replace("2, 2, 0", "2, 1, 1"), 2)


def test_real_bytecode_roundtrip_without_stage_metadata():
    text = '''module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
      func.func @probe(%a: tensor<8xf32>, %b: tensor<8xi32>,
                       %c: tensor<8xf32>, %d: tensor<8xi32>) {
        %r:2 = "hivm.hir.custom"(%a, %b, %c, %d)
          <{name = "flagtree.roundtrip_test", no_side_effect,
            operandSegmentSizes = array<i32: 2, 2, 0>}>
          {hivm.pipe = #hivm.pipe<PIPE_V>,
           hivm.tcore_type = #hivm.tcore_type<VECTOR>,
           hivm.vf_mode = #hivm.vf_mode<SIMD>, max_rank = 1 : i64,
           symbol = "custom_roundtrip_test"}
          : (tensor<8xf32>, tensor<8xi32>, tensor<8xf32>, tensor<8xi32>)
            -> (tensor<8xf32>, tensor<8xi32>)
        return
      }
    }'''
    options = SimpleNamespace(debug=False)
    bytecode = compiler.linalg_to_bc_by_triton_mlir_opt(text, {}, options)
    # A cache hit on the preceding stage supplies no process-local flag.
    restored = compiler.bc_to_linalg_by_bishengir_opt(bytecode, {}, options)
    assert "hivm.hir.custom" in restored
    assert "cann_compat.custom" not in restored
    assert "custom_roundtrip_test" in restored


def test_late_private_abi_keeps_layouts_and_sync():
    ub = '#hivm.address_space<ub>'
    data = f'memref<16xf16, strided<[16], offset: 3>, {ub}>'
    index = f'memref<16xi32, {ub}>'
    op = ('    "hivm.hir.custom"(%a, %i, %b, %tmp) '
          '<{name = "flagtree.structured_gather", operandSegmentSizes = array<i32: 2, 2>}> '
          '{symbol = "custom_flagtree_stride_gather_half"} : '
          f'({data}, {index}, {data}, {index}) -> ()')
    text = '"builtin.module"() ({\n  // keep synchronization\n' + op + '\n}) : () -> ()\n'
    result = late.lower_private_ops(text)
    assert '"hivm.hir.custom"' not in result
    assert result.count('memref.cast') == 4
    assert f': {data} to memref<?xf16, strided<[?], offset: ?>, {ub}>' in result
    assert 'func.call @custom_flagtree_stride_gather_half' in result
    assert result.count('func.func private @custom_flagtree_stride_gather_half') == 1
    assert '// keep synchronization' in result
    # Repeated uses share a declaration, and pre-existing SSA names cannot collide.
    repeated = text.replace(op, op + '\n' + op).replace('// keep', '// %flagtree_abi_ keep')
    result = late.lower_private_ops(repeated)
    assert result.count('func.func private @custom_flagtree_stride_gather_half') == 1
    assert '%flagtree_abi__0' in result
    with pytest.raises(ValueError, match='rank-one UB'):
        late.lower_private_ops(text.replace('16xf16', '4x4xf16'))
    with pytest.raises(ValueError, match='rank-one UB'):
        late.lower_private_ops(text.replace('<ub>', '<gm>'))
    with pytest.raises(ValueError, match='disagree'):
        late.lower_private_ops(text.replace('_gather_half', '_gather_float'))
    with pytest.raises(ValueError, match='result-free'):
        late.lower_private_ops(text.replace('-> ()\n})', '-> (tensor<16xf16>)\n})'))


def test_late_unrelated_operations_unchanged():
    text = '"hivm.hir.custom"(%a) {symbol = "user_kernel"} : (tensor<8xf32>) -> ()'
    assert late.lower_private_ops(text) == text


def test_late_storage_conversion_abi():
    for symbol, elements in [('byte_to_word', ('i8', 'i16', 'f16')),
                             ('word_to_byte', ('i16', 'i8', 'f16'))]:
        types = ', '.join(f'memref<64x{e}, #hivm.address_space<ub>>' for e in elements)
        text = ('"builtin.module"() ({\n'
                '    "hivm.hir.custom"(%a, %b, %tmp) '
                f'{{symbol = "custom_flagtree_{symbol}"}} : ({types}) -> ()\n'
                '}) : () -> ()\n')
        result = late.lower_private_ops(text)
        assert result.count('memref.cast') == 3
        assert f'func.call @custom_flagtree_{symbol}' in result


def test_effective_dispatch_is_hashed_and_explicit_option_wins(monkeypatch):
    from triton.backends.compiler import GPUTarget
    backend = compiler.AscendBackend(GPUTarget('npu', 'Ascend910B4', 0))
    monkeypatch.setenv('TRITON_ALL_BLOCKS_PARALLEL', '0')
    off = backend.parse_options({})
    on_explicit = backend.parse_options({'enable_auto_blockify': True})
    monkeypatch.setenv('TRITON_ALL_BLOCKS_PARALLEL', '1')
    on = backend.parse_options({})
    off_explicit = backend.parse_options({'enable_auto_blockify': False})
    assert off.enable_auto_blockify is False and on.enable_auto_blockify is True
    assert off.hash() == off_explicit.hash()
    assert on.hash() == on_explicit.hash()
    assert off.hash() != on.hash()


def test_join_lowering_policy_is_validated_and_hashed():
    from triton.backends.compiler import GPUTarget
    backend = compiler.AscendBackend(GPUTarget('npu', 'Ascend910B4', 0))
    options = [backend.parse_options({'structured_join_lowering': value})
               for value in ['auto', 'native', 'packed']]
    assert len({option.hash() for option in options}) == 3
    with pytest.raises(ValueError, match='structured_join_lowering'):
        backend.parse_options({'structured_join_lowering': 'unknown'})


@pytest.mark.parametrize('missing', ['none', 'metadata', 'binary', 'empty_binary', 'directory'])
def test_kernel_cache_completeness(tmp_path, missing):
    from triton.runtime.cache import _is_complete_kernel_cache
    meta, binary = tmp_path/'kernel.json', tmp_path/'kernel.npubin'
    meta.write_text('{}')
    binary.write_bytes(b'ELF')
    group = {'kernel.json': str(meta), 'kernel.npubin': str(binary)}
    if missing == 'metadata': meta.unlink()
    elif missing == 'binary': binary.unlink()
    elif missing == 'empty_binary': binary.write_bytes(b'')
    elif missing == 'directory':
        binary.unlink()
        binary.mkdir()
    assert _is_complete_kernel_cache(group, 'kernel.json', 'kernel.npubin') == (missing == 'none')
