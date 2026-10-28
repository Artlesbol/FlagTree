"""Host checks for physical UB address-table reuse and its rejection cases."""
import importlib
from pathlib import Path
import subprocess

import pytest

compiler = importlib.import_module('triton.backends.ascend.compiler')


@pytest.fixture(scope='module', autouse=True)
def assign_npu():
    yield


def test_physical_address_hoist_admission():
    source = Path(__file__).resolve().parents[1] / 'Conversion/structured_address_hoist.mlir'
    result = subprocess.check_output([
        compiler._get_triton_mlir_opt_path(), str(source),
        '--hoist-structured-permutation-addresses', '--allow-unregistered-dialect',
        '--verify-each'], text=True)
    admitted = {'packed_double', 'packed_single', 'unaligned_source', 'outside_loop_read_before'}
    rejected = {'index_extra_read', 'physical_alias', 'unknown_alias', 'index_write',
                'after_loop_read', 'strided_indices', 'unaligned_indices',
                'unaligned_scratch', 'unaligned_destination', 'overlapping_tables',
                'source_destination_alias', 'conditional_gather', 'unknown_source'}
    for name in admitted | rejected:
        body = result.split('func.func @' + name + '(')[1].split('\n  }')[0]
        if name in admitted:
            assert body.count('call @custom_flagtree_prepare_addresses') == 1, name
            assert body.count('call @custom_flagtree_addressed_gather_float') == 1, name
            assert body.index('prepare_addresses') < body.index('scf.for'), name
            assert 'call @custom_flagtree_stride_gather_float' not in body, name
        else:
            assert 'call @custom_flagtree_prepare_addresses' not in body, name
            assert body.count('call @custom_flagtree_stride_gather_float') == 1, name


def test_address_hoist_option_is_hashed():
    from triton.backends.compiler import GPUTarget
    backend = compiler.AscendBackend(GPUTarget('npu', 'Ascend910B4', 0))
    default = backend.parse_options({})
    enabled = backend.parse_options({'structured_address_hoisting': True})
    disabled = backend.parse_options({'structured_address_hoisting': False})
    assert default.structured_address_hoisting
    assert default.hash() == enabled.hash() and default.hash() != disabled.hash()


@pytest.mark.parametrize('value', [0, 1, 'true', None])
def test_address_hoist_rejects_non_boolean(value):
    with pytest.raises(ValueError, match='must be a bool'):
        compiler.AscendOptions(structured_address_hoisting=value)
