cmake_minimum_required(VERSION 3.20)
project(structured_permute LANGUAGES C CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-9.0.0" CACHE PATH "CANN toolkit")
set(SOC_VERSION "Ascend910B4" CACHE STRING "Target SoC")
set(RUN_MODE "npu")
include("${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
ascendc_library(permute_device STATIC aiv.cpp)
add_library(permute_reference SHARED launch.cpp)
target_link_libraries(permute_reference PRIVATE permute_device)

# Opt-in ABI contract tests link the same packaged device bitcode as Triton.
set(FLAGTREE_DEVICE_BITCODE "" CACHE FILEPATH "Packaged custom_ops.bc for UB ABI tests")
if(FLAGTREE_DEVICE_BITCODE)
  if(NOT EXISTS "${FLAGTREE_DEVICE_BITCODE}")
    message(FATAL_ERROR "Missing FLAGTREE_DEVICE_BITCODE: ${FLAGTREE_DEVICE_BITCODE}")
  endif()
  ascendc_library(ub_contract_device STATIC ub_contract.cpp)
  ascendc_compile_options(ub_contract_device PRIVATE
    "SHELL:-Xclang -mlink-builtin-bitcode -Xclang ${FLAGTREE_DEVICE_BITCODE}"
    --cce-auto-sync=off)
  add_library(ub_contract_reference SHARED ub_contract_launch.cpp)
  target_link_libraries(ub_contract_reference PRIVATE ub_contract_device)
endif()
