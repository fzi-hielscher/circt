//===- FoldTriggers.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include <string>

#define DEBUG_TYPE "arc-fold-triggers"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_FOLDTRIGGERS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace mlir;
using namespace arc;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct FoldTriggersPass : public arc::impl::FoldTriggersBase<FoldTriggersPass> {
  using arc::impl::FoldTriggersBase<FoldTriggersPass>::FoldTriggersBase;
  void runOnOperation() override;

  LogicalResult runOnModule(hw::HWModuleOp moduleOp);
  LogicalResult convertSequence(RewriterBase &rewriter,
                                sim::TriggerSequenceOp sequenceOp);
};
} // namespace

void FoldTriggersPass::runOnOperation() {
  for (auto moduleOp : getOperation().getOps<hw::HWModuleOp>())
    if (failed(runOnModule(moduleOp)))
      signalPassFailure();
}

static sim::OnEdgeOp getRootClockFromTrigger(Value trigger) {
  if (!isa<sim::EdgeTriggerType>(trigger.getType()))
    return {};

  while (auto defOp = trigger.getDefiningOp()) {
    if (auto rootOp = dyn_cast<sim::OnEdgeOp>(defOp)) {
      if (rootOp.getEvent() != hw::EventControl::AtPosEdge) {
        defOp->emitError("Non-posedge triggers are currently unsupported.");
        return {};
      }
      return rootOp;
    }
    auto sequenceOp = dyn_cast<sim::TriggerSequenceOp>(defOp);
    if (!sequenceOp) {
      defOp->emitError("Unsupported trigger operation.");
      return {};
    }
    trigger = sequenceOp.getParent();
  }
  return {};
}

static void mergeConcurrentTokenLoops(RewriterBase &rewriter, Value trigger) {
  SmallVector<ImpureTokenLoopOp> loopOpsToMerge;
  SmallVector<Location> locs;
  SmallVector<Value> exitTokens;

  for (auto &use : trigger.getUses()) {
    if (auto loopOp = dyn_cast<ImpureTokenLoopOp>(use.getOwner())) {
      loopOpsToMerge.emplace_back(loopOp);
      locs.push_back(loopOp.getLoc());
      exitTokens.push_back(loopOp.getExitToken());
    }
  }

  if (loopOpsToMerge.size() < 2)
    return;

  auto loc = FusedLoc::get(rewriter.getContext(), locs);
  rewriter.setInsertionPoint(loopOpsToMerge.front());
  auto joined = rewriter.createOrFold<TokenJoinOp>(loc, exitTokens);
  auto keptOp = loopOpsToMerge.front();
  rewriter.modifyOpInPlace(
      keptOp, [&]() { keptOp.getExitTokenMutable().assign(joined); });
  for (auto mergedLoop : ArrayRef(loopOpsToMerge).drop_front()) {
    rewriter.replaceAllUsesWith(mergedLoop.getEntryToken(),
                                keptOp.getEntryToken());
    rewriter.eraseOp(mergedLoop);
  }
}

LogicalResult
FoldTriggersPass::convertSequence(RewriterBase &rewriter,
                                  sim::TriggerSequenceOp sequenceOp) {
  if (sequenceOp.getNumResults() == 0) {
    rewriter.eraseOp(sequenceOp);
    return success();
  }

  for (auto res : sequenceOp.getResults())
    mergeConcurrentTokenLoops(rewriter, res);

  Value currentToken;
  ImpureTokenLoopOp keptLoop;
  for (auto [idx, res] : llvm::enumerate(sequenceOp.getResults())) {
    if (res.use_empty())
      continue;
    if (!res.hasOneUse()) {
      sequenceOp.emitError("Trigger at index " + Twine(idx) +
                           "could not be reduced.");
      return failure();
    }

    ImpureTokenLoopOp loop =
        dyn_cast<ImpureTokenLoopOp>(*res.getUsers().begin());
    if (!loop) {
      res.getUsers().begin()->emitOpError(" is not a supported trigger user.");
      return failure();
    }

    if (!keptLoop) {
      rewriter.modifyOpInPlace(loop, [&]() {
        loop.getInputMutable().assign(sequenceOp.getParent());
      });
      currentToken = loop.getExitToken();
      keptLoop = loop;
    } else {
      rewriter.replaceAllUsesWith(loop.getEntryToken(), currentToken);
      currentToken = loop.getExitToken();
      rewriter.eraseOp(loop);
    }
  }

  if (sequenceOp.getNumResults() > 1)
    rewriter.modifyOpInPlace(keptLoop, [&]() {
      keptLoop.getExitTokenMutable().assign(currentToken);
    });

  rewriter.eraseOp(sequenceOp);
  return success();
}

LogicalResult FoldTriggersPass::runOnModule(hw::HWModuleOp moduleOp) {
  SmallVector<sim::TriggerSequenceOp> leafSequences;
  std::optional<arc::InitialPseudoClockOp> initClockOp;
  SmallVector<Value> rootTriggers;
  SmallVector<sim::OnInitOp> onInitOps;

  IRRewriter rewriter(moduleOp);

  auto getInitClock = [&]() -> InitialPseudoClockOp {
    if (!initClockOp) {
      OpBuilder builder(moduleOp);
      builder.setInsertionPointToStart(moduleOp.getBodyBlock());
      initClockOp = builder.create<InitialPseudoClockOp>(moduleOp.getLoc());
    }
    return *initClockOp;
  };

  auto isLeafSequence = [](sim::TriggerSequenceOp sequenceOp) -> bool {
    if (!sequenceOp)
      return false;
    for (auto res : sequenceOp.getResults())
      for (auto &use : res.getUses())
        if (isa<sim::TriggerSequenceOp>(use.getOwner()))
          return false;
    return true;
  };

  bool hasFailed = false;
  for (auto &op : llvm::make_early_inc_range(moduleOp.getOps())) {

    // Resolve GetClockFromTriggerOps as we go.
    if (auto clockFromTriggerOp = dyn_cast<arc::GetClockFromTriggerOp>(op)) {
      if (isa<sim::InitTriggerType>(
              clockFromTriggerOp.getTrigger().getType())) {
        rewriter.replaceOp(clockFromTriggerOp, getInitClock().getResult());
      } else {
        auto clockOp = getRootClockFromTrigger(clockFromTriggerOp.getTrigger());
        if (!clockOp) {
          clockFromTriggerOp.emitError(
              "Unable to resolve trigger to root clock.");
          hasFailed = true;
        } else {
          rewriter.replaceOp(clockFromTriggerOp, clockOp.getClock());
        }
      }
      continue;
    }

    if (auto sequenceOp = dyn_cast<sim::TriggerSequenceOp>(op)) {
      if (isLeafSequence(sequenceOp))
        leafSequences.emplace_back(sequenceOp);
      continue;
    }

    if (auto onInitOp = dyn_cast<sim::OnInitOp>(op)) {
      rootTriggers.emplace_back(onInitOp.getResult());
      onInitOps.emplace_back(onInitOp);
      continue;
    }

    if (auto onEdgeOp = dyn_cast<sim::OnEdgeOp>(op)) {
      rootTriggers.emplace_back(onEdgeOp.getResult());
      continue;
    }
  }

  if (hasFailed)
    return failure();

  SmallVector<sim::TriggerSequenceOp> worklist;
  while (!leafSequences.empty()) {
    worklist = std::move(leafSequences);
    leafSequences = SmallVector<sim::TriggerSequenceOp>();
    for (auto sequence : worklist) {
      auto parentSeq =
          sequence.getParent().getDefiningOp<sim::TriggerSequenceOp>();
      if (failed(convertSequence(rewriter, sequence)))
        return failure();
      if (isLeafSequence(parentSeq))
        leafSequences.push_back(parentSeq);
    }
  }

  for (auto root : rootTriggers)
    mergeConcurrentTokenLoops(rewriter, root);

  if (!onInitOps.empty()) {
    rewriter.setInsertionPoint(onInitOps.front());

    SmallVector<Location> locs;
    for (auto onInitOp : onInitOps)
      locs.push_back(onInitOp.getLoc());

    auto initClock = getInitClock();
    auto pseudoClockTrigger = rewriter.create<sim::OnEdgeOp>(
        FusedLoc::get(rewriter.getContext(), locs), initClock.getResult(),
        hw::EventControl::AtPosEdge);

    for (auto onInitOp : onInitOps) {
      assert(onInitOp.getResult().hasOneUse());
      rewriter.replaceAllOpUsesWith(onInitOp, {pseudoClockTrigger.getResult()});
      rewriter.eraseOp(onInitOp);
    }
  }

  return success();
}
