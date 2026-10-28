#include <aclrtlaunch_matrix_transport_aiv.h>
#include <aclrtlaunch_permute_aiv.h>
#include <cstdint>

extern "C" uint32_t launch_permute(uint32_t blocks, void *stream, void *x,
                                   void *z, void *y, void *indices, void *t,
                                   void *j, uint32_t q, uint32_t batch,
                                   uint32_t mode, uint32_t element_bytes) {
  return ACLRT_LAUNCH_KERNEL(permute_aiv)(
      blocks, stream, static_cast<uint8_t *>(x), static_cast<uint8_t *>(z),
      static_cast<uint8_t *>(y), static_cast<uint8_t *>(indices),
      static_cast<uint8_t *>(t), static_cast<uint8_t *>(j), q, batch, mode,
      element_bytes);
}

extern "C" uint32_t launch_matrix_transport(void *stream, void *x, void *y,
                                            void *indices, uint32_t transpose) {
  return ACLRT_LAUNCH_KERNEL(matrix_transport_aiv)(
      40, stream, static_cast<uint8_t *>(x), static_cast<uint8_t *>(y),
      static_cast<uint8_t *>(indices), transpose);
}
