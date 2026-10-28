// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Private compiler primitive: logical element indices, arbitrary UB strides.
#include "Utils.h"
#include "Vector/VecUtils.h"
#include "Vector/Arange/ArangeUtils.h"
#include <type_traits>

template <typename T>
__aiv__ __attribute__((always_inline)) void structured_gather(
    memref_t<__ubuf__ T, 1> *src,
    memref_t<__ubuf__ int32_t, 1> *indices,
    memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ int32_t, 1> *addresses) {
  using Bits = typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;
  const int64_t n = indices->sizes[0];
  auto src_ptr = reinterpret_cast<__ubuf__ Bits *>(src->aligned + src->offset);
  auto dst_ptr = reinterpret_cast<__ubuf__ Bits *>(dst->aligned + dst->offset);
  auto idx_ptr = indices->aligned + indices->offset;
  auto address_ptr = addresses->aligned + addresses->offset;

  // This branch also preserves holes in a strided destination. The 910B
  // vector ISA has no general scatter exposed by the CANN SDK. The common
  // packed case below has no data-length scalar loop, even for strided src.
  if (indices->strides[0] != 1 || addresses->strides[0] != 1 ||
      dst->strides[0] != 1 || !isAddress32ByteAligned(dst_ptr) ||
      !isAddress32ByteAligned(idx_ptr) || !isAddress32ByteAligned(address_ptr)) {
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    for (int64_t i = 0; i < n; ++i)
      dst_ptr[i * dst->strides[0]] =
          src_ptr[static_cast<int64_t>(idx_ptr[i * indices->strides[0]]) *
                  src->strides[0]];
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }

  // All arithmetic acts on logical indices; padding is never a data element.
  const int32_t scale = src->strides[0] * sizeof(T);
  const int32_t base = static_cast<int32_t>(reinterpret_cast<intptr_t>(src_ptr));
  vector_eltwise_vs_1d<VectorOpTy::VMULS>(indices, scale, addresses);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vs_1d<VectorOpTy::VADDS>(addresses, base, addresses);
  INTRINSIC(pipe_barrier, PIPE_V);
  auto addr_ptr = reinterpret_cast<__ubuf__ uint32_t *>(
      addresses->aligned + addresses->offset);
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, n);
  INTRINSIC(vgather, dst_ptr, addr_ptr, 0, 8, 1);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

#define REGISTER_STRUCTURED_GATHER(SUFFIX, TYPE)                                \
  extern "C" __aiv__ __attribute__((always_inline)) void                         \
      _mlir_ciface_custom_flagtree_stride_gather_##SUFFIX(                       \
          memref_t<__ubuf__ TYPE, 1> *src,                                      \
          memref_t<__ubuf__ int32_t, 1> *indices,                               \
          memref_t<__ubuf__ TYPE, 1> *dst,                                      \
          memref_t<__ubuf__ int32_t, 1> *addresses) {                           \
    structured_gather(src, indices, dst, addresses);                             \
  }

REGISTER_STRUCTURED_GATHER(half, half)
REGISTER_STRUCTURED_GATHER(bf16, bfloat16_t)
REGISTER_STRUCTURED_GATHER(float, float)
REGISTER_STRUCTURED_GATHER(int16, int16_t)
REGISTER_STRUCTURED_GATHER(int32, int32_t)

template <typename T>
__aiv__ __attribute__((always_inline)) void structured_split(
    memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *lhs,
    memref_t<__ubuf__ T, 1> *rhs, memref_t<__ubuf__ int32_t, 1> *addresses) {
  using Bits = typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;
  const int64_t n = lhs->sizes[0];
  auto in = reinterpret_cast<__ubuf__ Bits *>(src->aligned + src->offset);
  auto a = reinterpret_cast<__ubuf__ Bits *>(lhs->aligned + lhs->offset);
  auto b = reinterpret_cast<__ubuf__ Bits *>(rhs->aligned + rhs->offset);
  bool alignedOutputs = isAddress32ByteAligned(a) && isAddress32ByteAligned(b);
  if (src->strides[0] == 1 && lhs->strides[0] == 1 && rhs->strides[0] == 1 &&
      isAddress32ByteAligned(in) && alignedOutputs) {
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, 2 * n);
    INTRINSIC(vreducev2, a, in, in, 1, 1, 1, 8, 0);
    INTRINSIC(vreducev2, b, in, in, 1, 1, 2, 8, 0);
    INTRINSIC_NO_ARGS(set_mask_norm);
    return;
  }
  if (lhs->strides[0] == 1 && rhs->strides[0] == 1 && addresses->strides[0] == 1 &&
      alignedOutputs && isAddress32ByteAligned(addresses->aligned + addresses->offset)) {
    const int32_t scale = src->strides[0] * sizeof(T);
    const int32_t base = static_cast<int32_t>(reinterpret_cast<intptr_t>(in));
    auto addr = addresses->aligned + addresses->offset;
    arange_1d_core<int32_t>(addr, n, base, 2 * scale);
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, n);
    INTRINSIC(vgather, a, reinterpret_cast<__ubuf__ uint32_t *>(addr), 0, 8, 1);
    INTRINSIC_NO_ARGS(set_mask_norm);
    INTRINSIC(pipe_barrier, PIPE_V);
    vector_eltwise_vs_1d<VectorOpTy::VADDS>(addresses, scale, addresses);
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, n);
    INTRINSIC(vgather, b, reinterpret_cast<__ubuf__ uint32_t *>(addr), 0, 8, 1);
    INTRINSIC_NO_ARGS(set_mask_norm);
    return;
  }
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int64_t i = 0; i < n; ++i) {
    a[i * lhs->strides[0]] = in[2 * i * src->strides[0]];
    b[i * rhs->strides[0]] = in[(2 * i + 1) * src->strides[0]];
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

#define REGISTER_STRUCTURED_SPLIT(SUFFIX, TYPE)                                 \
  extern "C" __aiv__ __attribute__((always_inline)) void                         \
      _mlir_ciface_custom_flagtree_split_##SUFFIX(                              \
          memref_t<__ubuf__ TYPE, 1> *src,                                      \
          memref_t<__ubuf__ TYPE, 1> *lhs,                                      \
          memref_t<__ubuf__ TYPE, 1> *rhs,                                      \
          memref_t<__ubuf__ int32_t, 1> *addresses) {                           \
    structured_split(src, lhs, rhs, addresses);                                 \
  }

REGISTER_STRUCTURED_SPLIT(half, half)
REGISTER_STRUCTURED_SPLIT(bf16, bfloat16_t)
REGISTER_STRUCTURED_SPLIT(float, float)
REGISTER_STRUCTURED_SPLIT(int16, int16_t)
REGISTER_STRUCTURED_SPLIT(int32, int32_t)

// These conversions serve byte permutations only: word values are always in
// [-128,127]. Half represents that range exactly, so no saturation/overflow
// conversion or CANN's sub-block 2D cast is required.
template <bool EXTEND>
__aiv__ __attribute__((always_inline)) void storage_cast(
    memref_t<__ubuf__ typename std::conditional<EXTEND, int8_t, int16_t>::type, 1> *src,
    memref_t<__ubuf__ typename std::conditional<EXTEND, int16_t, int8_t>::type, 1> *dst,
    memref_t<__ubuf__ half, 1> *scratch) {
  auto in = src->aligned + src->offset;
  auto out = dst->aligned + dst->offset;
  auto tmp = scratch->aligned + scratch->offset;
  const int64_t n = dst->sizes[0];
  if (src->strides[0] != 1 || dst->strides[0] != 1 || scratch->strides[0] != 1 ||
      ((reinterpret_cast<intptr_t>(in) | reinterpret_cast<intptr_t>(out) |
        reinterpret_cast<intptr_t>(tmp)) & 31)) {
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    for (int64_t i = 0; i < n; ++i)
      out[i * dst->strides[0]] = in[i * src->strides[0]];
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, n);
  if constexpr (EXTEND)
    INTRINSIC(vconv_s82f16, tmp, in, 1, 1, 1, 8, 4);
  else
    INTRINSIC(vconv_s162f16r, tmp, in, 1, 1, 1, 8, 8);
  INTRINSIC(pipe_barrier, PIPE_V);
  if constexpr (EXTEND)
    INTRINSIC(vconv_f162s16z, out, tmp, 1, 1, 1, 8, 8);
  else
    INTRINSIC(vconv_f162s8z, out, tmp, 1, 1, 1, 4, 8);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_flagtree_byte_to_word(memref_t<__ubuf__ int8_t, 1> *src,
    memref_t<__ubuf__ int16_t, 1> *dst, memref_t<__ubuf__ half, 1> *scratch) {
  storage_cast<true>(src, dst, scratch);
}

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_flagtree_word_to_byte(memref_t<__ubuf__ int16_t, 1> *src,
    memref_t<__ubuf__ int8_t, 1> *dst, memref_t<__ubuf__ half, 1> *scratch) {
  storage_cast<false>(src, dst, scratch);
}

// Late compiler optimization: reuse the two existing private index buffers as
// absolute address tables. The compiler proves exclusive use and fixed packed
// physical layouts before selecting this ABI. No new UB storage is required.
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_flagtree_prepare_addresses(
    memref_t<__ubuf__ int32_t, 1> *table_a,
    memref_t<__ubuf__ int32_t, 1> *table_b,
    int32_t scale, int32_t base_a, int32_t base_b) {
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vs_1d<VectorOpTy::VMULS>(table_a, scale, table_b);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vs_1d<VectorOpTy::VADDS>(table_b, base_a, table_a);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vs_1d<VectorOpTy::VADDS>(table_b, base_b, table_b);
  INTRINSIC(pipe_barrier, PIPE_V);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void addressed_gather(
    memref_t<__ubuf__ int32_t, 1> *table, memref_t<__ubuf__ T, 1> *dst) {
  using Bits = typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;
  auto out = reinterpret_cast<__ubuf__ Bits *>(dst->aligned + dst->offset);
  auto addresses = reinterpret_cast<__ubuf__ uint32_t *>(table->aligned + table->offset);
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, dst->sizes[0]);
  INTRINSIC(vgather, out, addresses, 0, 8, 1);
  INTRINSIC_NO_ARGS(set_mask_norm);
}
#define REGISTER_ADDRESSED_GATHER(SUFFIX, TYPE)                                 \
  extern "C" __aiv__ __attribute__((always_inline)) void                         \
      _mlir_ciface_custom_flagtree_addressed_gather_##SUFFIX(                     \
          memref_t<__ubuf__ int32_t, 1> *table,                                  \
          memref_t<__ubuf__ TYPE, 1> *dst) { addressed_gather(table, dst); }
REGISTER_ADDRESSED_GATHER(half, half)
REGISTER_ADDRESSED_GATHER(bf16, bfloat16_t)
REGISTER_ADDRESSED_GATHER(float, float)
REGISTER_ADDRESSED_GATHER(int16, int16_t)
REGISTER_ADDRESSED_GATHER(int32, int32_t)
