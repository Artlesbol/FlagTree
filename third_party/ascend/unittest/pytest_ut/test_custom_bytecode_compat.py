"""CPU checks for the current-MLIR / installed-CANN transport boundary."""
from types import SimpleNamespace
import importlib

import pytest
import torch_npu  # registers torch.npu for the suite's device fixture

compiler = importlib.import_module("triton.backends.ascend.compiler")


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
