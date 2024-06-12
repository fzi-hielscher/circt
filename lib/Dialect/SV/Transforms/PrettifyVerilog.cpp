//===- PrettifyVerilog.cpp - Transformations to improve Verilog quality ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass contains elective transformations that improve the quality of
// SystemVerilog generated by the ExportVerilog library.  This pass is not
// compulsory: things that are required for ExportVerilog to be correct should
// be included as part of the ExportVerilog pass itself to make sure it is self
// contained.  This allows the ExportVerilog pass to be simpler.
//
// PrettifyVerilog is run prior to Verilog emission but must be aware of the
// options in LoweringOptions.  It shouldn't introduce invalid constructs that
// aren't present in the IR already: this isn't a general "raising" pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;

namespace circt {
namespace sv {
#define GEN_PASS_DEF_PRETTIFYVERILOG
#include "circt/Dialect/SV/SVPasses.h.inc"
} // namespace sv
} // namespace circt

//===----------------------------------------------------------------------===//
// PrettifyVerilogPass
//===----------------------------------------------------------------------===//

namespace {
struct PrettifyVerilogPass
    : public circt::sv::impl::PrettifyVerilogBase<PrettifyVerilogPass> {
  void runOnOperation() override;

private:
  void processPostOrder(Block &block);
  bool prettifyUnaryOperator(Operation *op);
  void sinkOrCloneOpToUses(Operation *op);
  void sinkExpression(Operation *op);
  void useNamedOperands(Operation *op, DenseMap<Value, Operation *> &pipeMap);

  bool splitStructAssignment(OpBuilder &builder, hw::StructType ty, Value dst,
                             Value src);
  bool splitArrayAssignment(OpBuilder &builder, hw::ArrayType ty, Value dst,
                            Value src);
  bool splitAssignment(OpBuilder &builder, Value dst, Value src);

  bool anythingChanged;
  LoweringOptions options;

  DenseSet<Operation *> toDelete;
};
} // end anonymous namespace

/// Return true if this is something that will get printed as a unary operator
/// by the Verilog printer.
static bool isVerilogUnaryOperator(Operation *op) {
  if (isa<comb::ParityOp>(op))
    return true;

  if (auto xorOp = dyn_cast<comb::XorOp>(op))
    return xorOp.isBinaryNot();

  if (auto icmpOp = dyn_cast<comb::ICmpOp>(op))
    return icmpOp.isEqualAllOnes() || icmpOp.isNotEqualZero();

  return false;
}

/// Helper to convert a value to a constant integer if it is one.
static std::optional<APInt> getInt(Value value) {
  if (auto cst = dyn_cast_or_null<hw::ConstantOp>(value.getDefiningOp()))
    return cst.getValue();
  return std::nullopt;
}

// Checks whether the destination and the source of an assignment are the same.
// However, the destination is a value with an inout type in SV form, while the
// source is built using ops from HWAggregates.
static bool isSelfWrite(Value dst, Value src) {
  if (dst == src)
    return true;

  auto *srcOp = src.getDefiningOp();
  auto *dstOp = dst.getDefiningOp();
  if (!srcOp || !dstOp)
    return false;

  return TypeSwitch<Operation *, bool>(srcOp)
      .Case<hw::StructExtractOp>([&](auto extract) {
        auto toField = dyn_cast<sv::StructFieldInOutOp>(dstOp);
        if (!toField)
          return false;
        if (toField.getFieldAttr() != extract.getFieldNameAttr())
          return false;
        return isSelfWrite(toField.getInput(), extract.getInput());
      })
      .Case<hw::ArrayGetOp>([&](auto get) {
        auto toGet = dyn_cast<sv::ArrayIndexInOutOp>(dstOp);
        if (!toGet || toGet.getIndex().getType() != get.getIndex().getType())
          return false;
        auto toIdx = getInt(toGet.getIndex());
        auto fromIdx = getInt(get.getIndex());
        if (!toIdx || !fromIdx || toIdx != fromIdx)
          return false;
        return isSelfWrite(toGet.getInput(), get.getInput());
      })
      .Case<sv::ReadInOutOp>([&](auto read) { return dst == read.getInput(); })
      .Default([&](auto srcOp) { return false; });
}

/// Split an assignment to a structure to writes of individual fields.
bool PrettifyVerilogPass::splitStructAssignment(OpBuilder &builder,
                                                hw::StructType ty, Value dst,
                                                Value src) {
  // Follow a chain of injects to find all fields overwritten and separate
  // them into a series of field updates instead of a whole-structure write.
  DenseMap<StringAttr, std::pair<Location, Value>> fields;
  while (auto inj = dyn_cast_or_null<hw::StructInjectOp>(src.getDefiningOp())) {
    // Inner injects are overwritten by outer injects.
    // Insert does not overwrite the store to be lowered.
    auto field = std::make_pair(inj.getLoc(), inj.getNewValue());
    fields.try_emplace(inj.getFieldNameAttr(), field);
    src = inj.getInput();
  }

  // The origin must be either the object itself (partial update)
  // or the whole object must be overwritten.
  if (!isSelfWrite(dst, src) && ty.getElements().size() != fields.size())
    return false;

  // Emit the field assignments in the order of their definition.
  for (auto &field : ty.getElements()) {
    const auto &name = field.name;

    auto it = fields.find(name);
    if (it == fields.end())
      continue;

    auto &[loc, value] = it->second;
    auto ref = builder.create<sv::StructFieldInOutOp>(loc, dst, name);
    if (!splitAssignment(builder, ref, value))
      builder.create<sv::PAssignOp>(loc, ref, value);
  }
  return true;
}

/// Split an assignment to an array element to writes of individual indices.
bool PrettifyVerilogPass::splitArrayAssignment(OpBuilder &builder,
                                               hw::ArrayType ty, Value dst,
                                               Value src) {
  // Follow a chain of concat + slice operations that alter a single element.
  if (auto op = dyn_cast_or_null<hw::ArrayCreateOp>(src.getDefiningOp())) {
    // TODO: consider breaking up array assignments into assignments
    // to individual fields.
    auto ty = hw::type_cast<hw::ArrayType>(op.getType());
    if (ty.getNumElements() != 1)
      return false;
    APInt zero(std::max(1u, llvm::Log2_64_Ceil(ty.getNumElements())), 0);

    Value value = op.getInputs()[0];
    auto loc = op.getLoc();
    auto index = builder.create<hw::ConstantOp>(loc, zero);

    auto field = builder.create<sv::ArrayIndexInOutOp>(loc, dst, index);
    if (!splitAssignment(builder, field, value))
      builder.create<sv::PAssignOp>(loc, field, value);
    return true;
  }

  // TODO: generalise to ranges and arbitrary concatenations.
  SmallVector<std::tuple<APInt, Location, Value>> fields;
  while (auto concat =
             dyn_cast_or_null<hw::ArrayConcatOp>(src.getDefiningOp())) {
    auto loc = concat.getLoc();

    // Look for a slice and an element:
    // concat(slice(a, 0, size - 1), elem)
    // concat(elem, slice(a, 1, size - 1))
    if (concat.getNumOperands() == 2) {
      auto c = concat.getInputs();

      auto lhs = dyn_cast_or_null<hw::ArraySliceOp>(c[1].getDefiningOp());
      auto rhs = dyn_cast_or_null<hw::ArraySliceOp>(c[0].getDefiningOp());
      auto midL = dyn_cast_or_null<hw::ArrayCreateOp>(c[1].getDefiningOp());
      auto midR = dyn_cast_or_null<hw::ArrayCreateOp>(c[0].getDefiningOp());

      auto size =
          hw::type_cast<hw::ArrayType>(concat.getType()).getNumElements();
      if (lhs && midR) {
        auto baseIdx = getInt(lhs.getLowIndex());
        if (!baseIdx || *baseIdx != 0 || midR.getInputs().size() != 1)
          break;
        fields.emplace_back(APInt(baseIdx->getBitWidth(), size - 1), loc,
                            midR.getInputs()[0]);
        src = lhs.getInput();
        continue;
      }
      if (rhs && midL) {
        auto baseIdx = getInt(rhs.getLowIndex());
        if (!baseIdx || *baseIdx != 1 || midL.getInputs().size() != 1)
          break;
        src = rhs.getInput();
        fields.emplace_back(APInt(baseIdx->getBitWidth(), 0), loc,
                            midL.getInputs()[0]);
        continue;
      }
      break;
    }

    // Look for a pattern overwriting a single element of the array.
    // concat(slice(a, 0, n - 1), create(get(a, n)), slice(n + 1, size -
    // n))
    if (concat.getNumOperands() == 3) {
      auto c = concat.getInputs();
      auto rhs = dyn_cast_or_null<hw::ArraySliceOp>(c[0].getDefiningOp());
      auto mid = dyn_cast_or_null<hw::ArrayCreateOp>(c[1].getDefiningOp());
      auto lhs = dyn_cast_or_null<hw::ArraySliceOp>(c[2].getDefiningOp());
      if (!lhs || !mid || !rhs || mid.getInputs().size() != 1)
        break;
      auto elem = mid.getInputs()[0];
      auto arr = lhs.getInput();
      if (arr != rhs.getInput() || arr.getType() != concat.getType())
        break;

      auto lhsSize =
          hw::type_cast<hw::ArrayType>(lhs.getType()).getNumElements();
      auto lhsIdx = getInt(lhs.getLowIndex());
      auto rhsIdx = getInt(rhs.getLowIndex());
      if (!lhsIdx || *lhsIdx != 0)
        break;
      if (!rhsIdx || *rhsIdx != lhsSize + 1)
        break;
      fields.emplace_back(*rhsIdx - 1, loc, elem);
      src = arr;
      continue;
    }
    break;
  }

  if (!isSelfWrite(dst, src))
    return false;

  // Emit the assignments in the order of the indices.
  std::stable_sort(fields.begin(), fields.end(), [](auto l, auto r) {
    return std::get<0>(l).ult(std::get<0>(r));
  });

  std::optional<APInt> last;
  for (auto &[i, loc, value] : fields) {
    if (i == last)
      continue;
    auto index = builder.create<hw::ConstantOp>(loc, i);
    auto field = builder.create<sv::ArrayIndexInOutOp>(loc, dst, index);
    if (!splitAssignment(builder, field, value))
      builder.create<sv::PAssignOp>(loc, field, value);
    last = i;
  }
  return true;
}

/// Instead of emitting a struct_inject to alter fields or concatenation
/// to adjust array elements, emit a more readable sequence of disjoint
/// assignments to individual fields and indices.  Returns true if
/// sub-assignments were emitted and the original one can be deleted.
bool PrettifyVerilogPass::splitAssignment(OpBuilder &builder, Value dst,
                                          Value src) {
  if (isSelfWrite(dst, src))
    return true;

  if (auto ty = hw::type_dyn_cast<hw::StructType>(src.getType()))
    return splitStructAssignment(builder, ty, dst, src);

  if (auto ty = hw::type_dyn_cast<hw::ArrayType>(src.getType()))
    return splitArrayAssignment(builder, ty, dst, src);

  return false;
}

/// Sink an operation into the same block where it is used.  This will clone the
/// operation so it can be sunk into multiple blocks. If there are no more uses
/// in the current block, the op will be removed.
void PrettifyVerilogPass::sinkOrCloneOpToUses(Operation *op) {
  assert(mlir::isMemoryEffectFree(op) &&
         "Op with side effects cannot be sunk to its uses.");
  auto block = op->getBlock();
  // This maps a block to the block local instance of the op.
  SmallDenseMap<Block *, Value, 8> blockLocalValues;
  for (auto &use : llvm::make_early_inc_range(op->getUses())) {
    // If the current use is in the same block as the operation, there is
    // nothing to do.
    auto localBlock = use.getOwner()->getBlock();
    if (block == localBlock)
      continue;
    // Find the block local clone of the operation. If there is not one already,
    // the op will be cloned in to the block.
    auto &localValue = blockLocalValues[localBlock];
    if (!localValue) {
      // Clone the operation and insert it to the beginning of the block.
      localValue = OpBuilder::atBlockBegin(localBlock).clone(*op)->getResult(0);
    }
    // Replace the current use, removing it from the use list.
    use.set(localValue);
    anythingChanged = true;
  }
  // If this op is no longer used, drop it.
  if (op->use_empty()) {
    toDelete.insert(op);
    anythingChanged = true;
  }
}

/// This is called on unary operators.  This returns true if the operator is
/// moved.
bool PrettifyVerilogPass::prettifyUnaryOperator(Operation *op) {
  // If this is a multiple use unary operator, duplicate it and move it into the
  // block corresponding to the user.  This avoids emitting a temporary just for
  // a unary operator.  Instead of:
  //
  //    tmp1 = ^(thing+thing);
  //         = tmp1 + 42
  //
  // we get:
  //
  //    tmp2 = thing+thing;
  //         = ^tmp2 + 42
  //
  // This is particularly helpful when the operand of the unary op has multiple
  // uses as well.
  if (op->use_empty() || op->hasOneUse())
    return false;

  // If this operation has any users that cannot inline the operation, then
  // don't duplicate any of them.
  for (auto *user : op->getUsers()) {
    if (isa<comb::ExtractOp, hw::ArraySliceOp>(user))
      return false;
    if (!options.allowExprInEventControl &&
        isa<sv::AlwaysFFOp, sv::AlwaysOp>(user))
      return false;
  }

  // Duplicating unary operations can move them across blocks (down the region
  // tree).  Make sure to keep referenced constants local.
  auto cloneConstantOperandsIfNeeded = [&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      auto constant = operand.get().getDefiningOp<hw::ConstantOp>();
      if (!constant)
        continue;

      // If the constant is in a different block, clone or move it into the
      // block.
      if (constant->getBlock() != op->getBlock())
        operand.set(OpBuilder(op).clone(*constant)->getResult(0));
    }
  };

  while (!op->hasOneUse()) {
    OpOperand &use = *op->use_begin();
    Operation *user = use.getOwner();

    // Clone the operation and insert before this user.
    auto *cloned = OpBuilder(user).clone(*op);
    cloneConstantOperandsIfNeeded(cloned);

    // Update user's operand to the new value.
    use.set(cloned->getResult(0));
  }

  // There is exactly one user left, so move this before it.
  Operation *user = *op->user_begin();
  op->moveBefore(user);
  cloneConstantOperandsIfNeeded(op);

  anythingChanged = true;
  return true;
}

// Return the depth of the specified block in the region tree, stopping at
// 'topBlock'.
static unsigned getBlockDepth(Block *block, Block *topBlock) {
  unsigned result = 0;
  while (block != topBlock) {
    block = block->getParentOp()->getBlock();
    ++result;
  }
  return result;
}

/// This method is called on expressions to see if we can sink them down the
/// region tree.  This is a good thing to do to reduce scope of the expression.
///
/// This is called on expressions that may have side effects, and filters out
/// the side effecting cases late for efficiency.
void PrettifyVerilogPass::sinkExpression(Operation *op) {
  // Ignore expressions with no users.
  if (op->use_empty())
    return;

  Block *curOpBlock = op->getBlock();

  // Single-used expressions are the most common and simple to handle.
  if (op->hasOneUse()) {
    if (curOpBlock != op->user_begin()->getBlock()) {
      // Ok, we're about to make a change, ensure that there are no side
      // effects.
      if (!mlir::isMemoryEffectFree(op))
        return;

      op->moveBefore(*op->user_begin());
      anythingChanged = true;
    }
    return;
  }

  // Find the nearest common ancestor of all the users.
  auto userIt = op->user_begin();
  Block *ncaBlock = userIt->getBlock();
  ++userIt;
  unsigned ncaBlockDepth = getBlockDepth(ncaBlock, curOpBlock);
  if (ncaBlockDepth == 0)
    return; // Have a user in the current block.

  for (auto e = op->user_end(); userIt != e; ++userIt) {
    auto *userBlock = userIt->getBlock();
    if (userBlock == curOpBlock)
      return; // Op has a user in it own block, can't sink it.
    if (userBlock == ncaBlock)
      continue;

    // Get the region depth of the user block so we can march up the region tree
    // to a common ancestor.
    unsigned userBlockDepth = getBlockDepth(userBlock, curOpBlock);
    while (userBlock != ncaBlock) {
      if (ncaBlockDepth < userBlockDepth) {
        userBlock = userBlock->getParentOp()->getBlock();
        --userBlockDepth;
      } else if (userBlockDepth < ncaBlockDepth) {
        ncaBlock = ncaBlock->getParentOp()->getBlock();
        --ncaBlockDepth;
      } else {
        userBlock = userBlock->getParentOp()->getBlock();
        --userBlockDepth;
        ncaBlock = ncaBlock->getParentOp()->getBlock();
        --ncaBlockDepth;
      }
    }

    if (ncaBlockDepth == 0)
      return; // Have a user in the current block.
  }

  // Ok, we're about to make a change, ensure that there are no side
  // effects.
  if (!mlir::isMemoryEffectFree(op))
    return;

  // Ok, we found a common ancestor between all the users that is deeper than
  // the current op.  Sink it into the start of that block.
  assert(ncaBlock != curOpBlock && "should have bailed out earlier");
  op->moveBefore(&ncaBlock->front());
  anythingChanged = true;
}

void PrettifyVerilogPass::processPostOrder(Block &body) {
  SmallVector<Operation *> instances;

  // Walk the block bottom-up, processing the region tree inside out.
  for (auto &op :
       llvm::make_early_inc_range(llvm::reverse(body.getOperations()))) {
    if (op.getNumRegions()) {
      for (auto &region : op.getRegions())
        for (auto &regionBlock : region.getBlocks())
          processPostOrder(regionBlock);
    }

    // Simplify assignments involving structures and arrays.
    if (auto assign = dyn_cast<sv::PAssignOp>(op)) {
      auto dst = assign.getDest();
      auto src = assign.getSrc();
      if (!isSelfWrite(dst, src)) {
        OpBuilder builder(assign);
        if (splitAssignment(builder, dst, src)) {
          anythingChanged = true;
          toDelete.insert(src.getDefiningOp());
          toDelete.insert(dst.getDefiningOp());
          assign.erase();
          continue;
        }
      }
    }

    // Sink and duplicate unary operators.
    if (isVerilogUnaryOperator(&op) && prettifyUnaryOperator(&op))
      continue;

    // Sink or duplicate constant ops and invisible "free" ops into the same
    // block as their use.  This will allow the verilog emitter to inline
    // constant expressions and avoids ReadInOutOp from preventing motion.
    if (matchPattern(&op, mlir::m_Constant()) ||
        isa<sv::ReadInOutOp, sv::ArrayIndexInOutOp, sv::StructFieldInOutOp,
            sv::IndexedPartSelectInOutOp, hw::ParamValueOp>(op)) {
      sinkOrCloneOpToUses(&op);
      continue;
    }

    // Sink normal expressions down the region tree if they aren't used within
    // their current block.  This allows them to be folded into the using
    // expression inline in the best case, and better scopes the temporary wire
    // they generate in the worst case.  Our overall traversal order is
    // post-order here which means all users will already be sunk.
    if (hw::isCombinational(&op) || sv::isExpression(&op)) {
      sinkExpression(&op);
      continue;
    }

    if (isa<hw::InstanceOp>(op))
      instances.push_back(&op);
  }

  // If we have any instances, keep their relative order but shift them to the
  // end of the module.  Any outputs will be writing to a wire or an output port
  // of the enclosing module anyway, and this allows inputs to be inlined into
  // the operand list as parameters.
  if (!instances.empty()) {
    for (Operation *instance : llvm::reverse(instances)) {
      if (instance != &body.back())
        instance->moveBefore(&body.back());
    }
  }
}

void PrettifyVerilogPass::runOnOperation() {
  hw::HWModuleOp thisModule = getOperation();
  options = LoweringOptions(thisModule->getParentOfType<mlir::ModuleOp>());

  // Keeps track if anything changed during this pass, used to determine if
  // the analyses were preserved.
  anythingChanged = false;

  // Walk the operations in post-order, transforming any that are interesting.
  processPostOrder(*thisModule.getBodyBlock());

  // Erase any dangling operands of simplified operations.
  while (!toDelete.empty()) {
    auto it = toDelete.begin();
    Operation *op = *it;
    toDelete.erase(it);

    if (!op || !isOpTriviallyDead(op))
      continue;

    for (auto operand : op->getOperands())
      toDelete.insert(operand.getDefiningOp());

    op->erase();
  }

  // If we did not change anything in the graph mark all analysis as
  // preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass> circt::sv::createPrettifyVerilogPass() {
  return std::make_unique<PrettifyVerilogPass>();
}
