// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include <limits>
#include <optional>

namespace {
// This pass runs after CANN memory planning. It never allocates or moves UB
// buffers: two existing, private index buffers become absolute address tables.
// All admission is completed before changing the IR.
struct PermuteBuffer {
  int64_t address, count, bytes;
  mlir::Attribute space;
};
struct PermuteSource {
  PermuteBuffer first, second;
  mlir::Value condition;
};

static mlir::LogicalResult getPermuteStrides(
    mlir::MemRefType type, llvm::SmallVectorImpl<int64_t> &strides,
    int64_t &offset) {
#ifdef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  return type.getStridesAndOffset(strides, offset);
#else
  return mlir::getStridesAndOffset(type, strides, offset);
#endif
}

static std::optional<PermuteBuffer> resolvePermuteBuffer(mlir::Value value) {
  using namespace mlir;
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return resolvePermuteBuffer(cast.getSource());
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return std::nullopt;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getPermuteStrides(type, strides, offset)) ||
      ShapedType::isDynamic(offset))
    return std::nullopt;
  int64_t count = 1;
  for (int i = type.getRank() - 1; i >= 0; --i) {
    if (type.getDimSize(i) != 1 && strides[i] != count)
      return std::nullopt;
    if (type.getDimSize(i) <= 0 || count > 4096 / type.getDimSize(i))
      return std::nullopt;
    count *= type.getDimSize(i);
  }
  int64_t bytes = type.getElementTypeBitWidth() / 8;
  if (bytes != 2 && bytes != 4)
    return std::nullopt;
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (op->getName().getStringRef() == "hivm.hir.pointer_cast" &&
      op->getNumOperands() == 1) {
    llvm::APInt address;
    if (!matchPattern(op->getOperand(0), m_ConstantInt(&address)))
      return std::nullopt;
    int64_t base = address.getSExtValue();
    if (base < 0 || base > std::numeric_limits<int32_t>::max() || offset < 0 ||
        offset > (std::numeric_limits<int32_t>::max() - base) / bytes)
      return std::nullopt;
    int64_t start = base + offset * bytes;
    if (start > std::numeric_limits<int32_t>::max() - count * bytes)
      return std::nullopt;
    return PermuteBuffer{start, count, bytes, type.getMemorySpace()};
  }
  if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp>(op)) {
    auto source = resolvePermuteBuffer(op->getOperand(0));
    if (source && source->count == count && source->bytes == bytes)
      return source;
  }
  if (auto view = dyn_cast<memref::SubViewOp>(op)) {
    auto source = resolvePermuteBuffer(view.getSource());
    auto sourceType = cast<MemRefType>(view.getSource().getType());
    SmallVector<int64_t> sourceStrides;
    int64_t sourceOffset;
    if (!source || failed(getPermuteStrides(sourceType, sourceStrides, sourceOffset)))
      return std::nullopt;
    int64_t start = source->address;
    for (auto [i, off] : llvm::enumerate(view.getStaticOffsets())) {
      if (ShapedType::isDynamic(off) || off < 0 || off >= sourceType.getDimSize(i))
        return std::nullopt;
      start += off * sourceStrides[i] * bytes;
    }
    if (start < source->address || start + count * bytes >
                                      source->address + source->count * bytes)
      return std::nullopt;
    return PermuteBuffer{start, count, bytes, source->space};
  }
  return std::nullopt;
}

static std::optional<PermuteSource> resolvePermuteSource(mlir::Value value, bool aliasOnly = false) {
  using namespace mlir;
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return resolvePermuteSource(cast.getSource(), aliasOnly);
  if (auto *op = value.getDefiningOp()) {
    if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp>(op))
      return resolvePermuteSource(op->getOperand(0), aliasOnly);
    // For interference checks, a subview is conservatively covered by the
    // entire source allocation. Actual gather admission still needs an exact
    // packed descriptor; never use this over-approximation to generate indices.
    if (aliasOnly && isa<memref::SubViewOp>(op))
      return resolvePermuteSource(op->getOperand(0), true);
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      auto a = resolvePermuteBuffer(select.getTrueValue());
      auto b = resolvePermuteBuffer(select.getFalseValue());
      if (a && b && a->count == b->count && a->bytes == b->bytes &&
          a->space == b->space)
        return PermuteSource{*a, *b, select.getCondition()};
      return std::nullopt;
    }
  }
  if (auto buffer = resolvePermuteBuffer(value))
    return PermuteSource{*buffer, *buffer, {}};
  return std::nullopt;
}

static bool permuteOverlap(PermuteBuffer a, PermuteBuffer b) {
  return a.space == b.space && a.address < b.address + b.count * b.bytes &&
         b.address < a.address + a.count * a.bytes;
}

struct HoistStructuredPermutationAddressesPass
    : mlir::PassWrapper<HoistStructuredPermutationAddressesPass,
                       mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistStructuredPermutationAddressesPass)
  llvm::StringRef getArgument() const final {
    return "hoist-structured-permutation-addresses";
  }
  llvm::StringRef getDescription() const final {
    return "Reuse private vector permutation address tables across planned UB loops";
  }
  void runOnOperation() final {
    using namespace mlir;
    SmallVector<func::CallOp> candidates;
    getOperation().walk([&](func::CallOp call) {
      if (call.getCallee().starts_with("custom_flagtree_stride_gather_"))
        candidates.push_back(call);
    });
    for (auto call : candidates)
      optimize(call);
  }
  void optimize(mlir::func::CallOp call) {
    using namespace mlir;
    StringRef suffix = call.getCallee().drop_front(StringRef("custom_flagtree_stride_gather_").size());
    if (suffix != "half" && suffix != "bf16" && suffix != "float" &&
        suffix != "int16" && suffix != "int32")
      return;
    auto loop = call->getParentOfType<scf::ForOp>();
    auto function = call->getParentOfType<func::FuncOp>();
    if (!loop || !function || function.getBody().empty() ||
        loop->getBlock() != &function.getBody().front() ||
        call->getBlock() != loop.getBody() || call.getNumOperands() != 4 ||
        call.getNumResults() != 0)
      return;
    auto source = resolvePermuteSource(call.getOperand(0));
    auto indices = resolvePermuteBuffer(call.getOperand(1));
    auto destination = resolvePermuteBuffer(call.getOperand(2));
    auto scratch = resolvePermuteBuffer(call.getOperand(3));
    auto ub = hivm::AddressSpaceAttr::get(&getContext(), hivm::AddressSpace::UB);
    if (!source || !indices || !destination || !scratch || indices->space != ub ||
        indices->bytes != 4 || scratch->bytes != 4 ||
        source->first.bytes != destination->bytes ||
        call.getOperand(1).getType() != call.getOperand(3).getType() ||
        indices->count != source->first.count ||
        indices->count != destination->count || indices->count != scratch->count ||
        indices->space != source->first.space || indices->space != destination->space ||
        indices->space != scratch->space ||
        (indices->address | scratch->address | destination->address) % 32 ||
        permuteOverlap(*indices, *scratch))
      return;
    for (auto buffer : {source->first, source->second, *destination})
      if (permuteOverlap(*indices, buffer) || permuteOverlap(*scratch, buffer))
        return;
    // Source/destination overlap is deliberately not admitted by this path.
    if (permuteOverlap(source->first, *destination) ||
        permuteOverlap(source->second, *destination))
      return;
    // Any additional use of either table after preparation invalidates reuse.
    // Examine physical aliases too, not merely SSA uses of the original views.
    bool safe = true;
    function.walk([&](Operation *op) {
      if (!safe || op == function.getOperation() || op == call.getOperation() ||
          isa<memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
              memref::SubViewOp>(op))
        return;
      Operation *top = op;
      while (top->getBlock() != loop->getBlock() && top->getParentOp())
        top = top->getParentOp();
      if (top->getBlock() != loop->getBlock()) {
        safe = false;
        return;
      }
      if (top != loop.getOperation() && top->isBeforeInBlock(loop))
        return;
      for (Value value : op->getOperands()) {
        auto type = dyn_cast<MemRefType>(value.getType());
        if (!type || type.getMemorySpace() != indices->space)
          continue;
        auto buffer = resolvePermuteBuffer(value);
        if (buffer) {
          if (permuteOverlap(*buffer, *indices) || permuteOverlap(*buffer, *scratch))
            safe = false;
        } else if (auto alternatives = resolvePermuteSource(value, true)) {
          for (auto b : {alternatives->first, alternatives->second})
            if (permuteOverlap(b, *indices) || permuteOverlap(b, *scratch))
              safe = false;
        } else {
          safe = false;
        }
      }
    });
    if (!safe)
      return;

    OpBuilder builder(loop);
    auto loc = call.getLoc();
    auto module = getOperation();
    auto tableType = cast<MemRefType>(call.getOperand(1).getType());
    StringRef prepare = "custom_flagtree_prepare_addresses";
    std::string name = "custom_flagtree_addressed_gather_" + suffix.str();
    auto compatible = [&](StringRef symbol, TypeRange inputs) {
      auto *existing = module.lookupSymbol(symbol);
      if (!existing)
        return true;
      auto fn = dyn_cast<func::FuncOp>(existing);
      return fn && fn.isPrivate() && fn.getBody().empty() &&
             fn->hasAttr("flagtree.structured_address_table") &&
             fn.getFunctionType() == builder.getFunctionType(inputs, {});
    };
    if (!compatible(prepare, {tableType, tableType, builder.getI32Type(),
                              builder.getI32Type(), builder.getI32Type()}) ||
        !compatible(name, {tableType, call.getOperand(2).getType()}))
      return;
    auto staticTable = MemRefType::get({indices->count}, builder.getI32Type(), MemRefLayoutAttrInterface{}, indices->space);
    auto pointer = [&](int64_t address) -> Value {
      Value constant = builder.create<arith::ConstantIntOp>(loc, address, 64);
      Value storage = builder.create<hivm::PointerCastOp>(loc, staticTable, constant);
      return builder.create<memref::CastOp>(loc, tableType, storage);
    };
    Value tableA = pointer(indices->address), tableB = pointer(scratch->address);
    auto declare = [&](StringRef name, TypeRange types) {
      if (module.lookupSymbol(name))
        return;
      OpBuilder decl(module.getBodyRegion());
      auto fn = decl.create<func::FuncOp>(loc, name, builder.getFunctionType(types, {}));
      fn.setPrivate();
      fn->setAttr("hacc.always_inline", builder.getUnitAttr());
      fn->setAttr("flagtree.structured_address_table", builder.getUnitAttr());
      fn->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
      // Inherit the core-kind attribute from the existing private declaration.
      if (auto old = module.lookupSymbol<func::FuncOp>(call.getCallee()))
        if (auto kind = old->getAttr("hivm.func_core_type"))
          fn->setAttr("hivm.func_core_type", kind);
    };
    declare(prepare, {tableType, tableType, builder.getI32Type(), builder.getI32Type(), builder.getI32Type()});
    SmallVector<Value> args{tableA, tableB};
    for (int64_t v : {source->first.bytes, source->first.address, source->second.address})
      args.push_back(builder.create<arith::ConstantIntOp>(loc, v, 32));
    builder.create<func::CallOp>(loc, prepare, TypeRange{}, args);
    builder.setInsertionPoint(call);
    Value chosen = tableA;
    if (source->condition)
      chosen = builder.create<arith::SelectOp>(loc, source->condition, tableA, tableB);
    declare(name, {tableType, call.getOperand(2).getType()});
    builder.create<func::CallOp>(loc, name, TypeRange{}, ValueRange{chosen, call.getOperand(2)});
    call.erase();
  }
};
} // namespace

namespace mlir::triton::ascend {
void registerStructuredPermutationPasses() {
  mlir::PassRegistration<HoistStructuredPermutationAddressesPass>();
}
} // namespace mlir::triton::ascend
