// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Private compiler primitive: logical element indices, arbitrary UB strides.
#include "Utils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

template <typename T>
__aiv__ __attribute__((always_inline)) void structured_gather(
    memref_t<__ubuf__ T, 1> *src,
    memref_t<__ubuf__ int32_t, 1> *indices,
    memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ int32_t, 1> *addresses,
    memref_t<__ubuf__ T, 1> *scratch) {
  using Bits = typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;
  const int64_t n = indices->sizes[0];
  auto src_ptr = reinterpret_cast<__ubuf__ Bits *>(src->aligned + src->offset);
  auto dst_ptr = reinterpret_cast<__ubuf__ Bits *>(dst->aligned + dst->offset);
  auto idx_ptr = indices->aligned + indices->offset;

  // This branch also preserves holes in a strided destination. The 910B
  // vector ISA has no general scatter exposed by the CANN SDK. The common
  // packed case below has no data-length scalar loop, even for strided src.
  if (indices->strides[0] != 1 || addresses->strides[0] != 1 ||
      scratch->strides[0] != 1 || dst->strides[0] != 1) {
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
          memref_t<__ubuf__ int32_t, 1> *addresses,                             \
          memref_t<__ubuf__ TYPE, 1> *scratch) {                                \
    structured_gather(src, indices, dst, addresses, scratch);                    \
  }

REGISTER_STRUCTURED_GATHER(half, half)
REGISTER_STRUCTURED_GATHER(bf16, bfloat16_t)
REGISTER_STRUCTURED_GATHER(float, float)
REGISTER_STRUCTURED_GATHER(int16, int16_t)
REGISTER_STRUCTURED_GATHER(int32, int32_t)
