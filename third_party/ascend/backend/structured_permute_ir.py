"""Late ABI lowering for FlagTree's private, already bufferized UB operations.

Input must be the installed CANN optimizer's generic assembly. No layout,
allocation or synchronization is changed here: only the private operation's
calling convention is expanded, after CANN has scheduled it as a PIPE_V op.
"""
import re


def split_fields(text):
    fields, stack, start = [], [], 0
    quoted = escaped = False
    pairs = {')': '(', ']': '[', '}': '{', '>': '<'}
    for i, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in '([{<':
            stack.append(char)
        elif char in ')]}>' and not (char == '>' and i and text[i - 1] == '-'):
            if not stack or stack.pop() != pairs[char]:
                raise ValueError('Unbalanced MLIR assembly')
        elif char == ',' and not stack:
            fields.append(text[start:i].strip())
            start = i + 1
    if quoted or stack:
        raise ValueError('Unbalanced MLIR assembly')
    return fields + [text[start:].strip()]


def _canonical_memref(typ):
    if not typ.startswith('memref<') or not typ.endswith('>'):
        raise ValueError('Structured permutation must be bufferized before ABI lowering')
    fields = split_fields(typ[7:-1])
    match = re.fullmatch(r'(?:\d+|\?)x(f16|bf16|f32|i8|i16|i32)', fields[0])
    if not match or fields[-1] != '#hivm.address_space<ub>':
        raise ValueError('Expected a rank-one UB memref, got ' + typ)
    if len(fields) not in (2, 3) or (len(fields) == 3 and not fields[1].startswith('strided<')):
        raise ValueError('Unsupported structured permutation memref layout: ' + typ)
    return f'memref<?x{match[1]}, strided<[?], offset: ?>, #hivm.address_space<ub>>'


def lower_private_ops(text):
    """Expand strictly recognized private operations; reject malformed ABI input."""
    prefix = 'flagtree_abi_'
    while '%' + prefix in text:
        prefix += '_'
    declarations, output, serial = {}, [], 0
    for line in text.splitlines():
        if '"hivm.hir.custom"' not in line or 'custom_flagtree_' not in line:
            output.append(line)
            continue
        match = re.fullmatch(
            r'(\s*)"hivm\.hir\.custom"\(([^)]*)\)(.*) : \((.*)\) -> \(\)\s*', line)
        if not match:
            raise ValueError('Expected a result-free, bufferized private custom operation')
        indent, operands, attrs, types = match.groups()
        symbol = re.search(r'\bsymbol = "(custom_flagtree_(?:(?:stride_gather|split)_'
                           r'(?:half|bf16|float|int16|int32)|byte_to_word|word_to_byte))"', attrs)
        if not symbol:
            raise ValueError('Unknown FlagTree structured permutation ABI')
        name = symbol[1]
        operands, types = split_fields(operands), split_fields(types)
        cast = name.endswith(('byte_to_word', 'word_to_byte'))
        arity = 3 if cast else 4
        if len(operands) != arity or len(types) != arity or any(
                not re.fullmatch(r'%[\w.$-]+(?:#\d+)?', v) for v in operands):
            raise ValueError('Invalid structured permutation operands')
        canonical = [_canonical_memref(t) for t in types]
        if cast:
            expected = ['i8', 'i16', 'f16'] if name.endswith('byte_to_word') else ['i16', 'i8', 'f16']
        else:
            elem = dict(half='f16', bf16='bf16', float='f32', int16='i16', int32='i32')[name.rsplit('_', 1)[1]]
            expected = [elem, 'i32', elem, 'i32'] if 'stride_gather' in name else [elem]*3 + ['i32']
        if any(not t.startswith('memref<?x' + e + ',') for t, e in zip(canonical, expected)):
            raise ValueError('Private operation symbol and operand types disagree')
        if name in declarations and declarations[name] != canonical:
            raise ValueError('Inconsistent structured permutation declaration')
        declarations[name] = canonical
        casted = []
        for operand, typ, target in zip(operands, types, canonical):
            value = f'%{prefix}{serial}'
            serial += 1
            output.append(f'{indent}{value} = memref.cast {operand} : {typ} to {target}')
            casted.append(value)
        output.append(f'{indent}func.call @{name}({", ".join(casted)}) : '
                      f'({", ".join(canonical)}) -> ()')
    if not declarations:
        return text
    # Generic printing emits the root module's region terminator at column 0.
    roots = [i for i, line in enumerate(output) if line.startswith('})')]
    if len(roots) != 1:
        raise ValueError('Expected one generic top-level module')
    decls = []
    for name, types in declarations.items():
        if re.search(r'\bsym_name = "' + re.escape(name) + '"', text):
            raise ValueError('Private structured permutation symbol already declared')
        decls.append(f'  func.func private @{name}({", ".join(types)}) attributes '
                     '{hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, '
                     'llvm.emit_c_interface}')
    output[roots[0]:roots[0]] = decls
    return '\n'.join(output) + '\n'
