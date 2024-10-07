//===- ProceduralizeSim.cpp - Conversion to procedural operations ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform non-procedural simulation operations with clock and enable to
// procedural operations wrapped in a procedural region.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "circt/Support/Debug.h"

#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "proceduralize-sim"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_PROCEDURALIZESIM
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace llvm;
using namespace circt;
using namespace sim;

namespace {
struct ProceduralizeSimPass : impl::ProceduralizeSimBase<ProceduralizeSimPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult proceduralize(PrintFormattedOp printOp);
  LogicalResult proceduralize(DPICallOp callOp);
  void proceduralize(FinishOp finishOp);

  SmallVector<Operation *> getPrintTokens(PrintFormattedOp op);
  void cleanup();

  // List of formatting ops to be pruned after proceduralization.
  SmallVector<Operation *> cleanupList;
};
} // namespace

LogicalResult ProceduralizeSimPass::proceduralize(PrintFormattedOp printOp) {
  // Get the flat list of formatting tokens and collect leaf tokens
  SmallVector<Value> flatString;
  if (auto concatInput =
          printOp.getInput().getDefiningOp<FormatStringConcatOp>()) {

    auto isAcyclic = concatInput.getFlattenedInputs(flatString);
    if (failed(isAcyclic)) {
      printOp.emitError("Cyclic format string cannot be proceduralized.");
      return failure();
    }
  } else {
    flatString.push_back(printOp.getInput());
  }

  SmallVector<Operation *> tokenList;
  SmallSetVector<Value, 4> arguments;
  for (auto &token : flatString) {
    auto *fmtOp = token.getDefiningOp();
    if (!fmtOp) {
      printOp.emitError("Proceduralization of format strings passed as block "
                        "argument is unsupported.");
      return failure();
    }
    tokenList.push_back(fmtOp);
    // For non-literal tokens, the value to be formatted has to become an
    // argument.
    if (!llvm::isa<FormatLitOp>(fmtOp)) {
      auto fmtVal = getFormattedValue(fmtOp);
      assert(!!fmtVal && "Unexpected foramtting token op.");
      arguments.insert(fmtVal);
    }
  }

  ImplicitLocOpBuilder builder(printOp.getLoc(), printOp);

  SmallVector<Value> argVec = arguments.takeVector();
  SmallVector<Type> argTypes;
  for (auto arg : argVec)
    argTypes.emplace_back(arg.getType());

  auto procOp = builder.create<sim::TriggeredOp>(
      TypeRange{}, printOp.getTrigger(), printOp.getCondition(),
      ValueRange{argVec}, ArrayAttr{});
  auto body = builder.createBlock(&procOp.getBody());
  SmallVector<Location> locs(argTypes.size(), builder.getLoc());
  procOp.getBody().addArguments(argTypes, locs);

  // Map the collected arguments to the newly created block arguments.
  IRMapping argumentMapper;
  unsigned idx = 0;
  for (auto arg : argVec) {
    argumentMapper.map(arg, procOp.getBody().getArgument(idx));
    idx++;
  }

  SmallDenseMap<Operation *, Operation *> cloneMap;

  // Create a copy of the required token operations within the TriggeredOp's
  // body.
  SmallVector<Value> clonedOperands;
  builder.setInsertionPointToStart(body);
  for (auto *token : tokenList) {
    auto &fmtCloned = cloneMap[token];
    if (!fmtCloned)
      fmtCloned = builder.clone(*token, argumentMapper);
    clonedOperands.push_back(fmtCloned->getResult(0));
  }
  // Concatenate tokens to a single value if necessary.
  Value procPrintInput;
  if (clonedOperands.size() != 1)
    procPrintInput = builder.createOrFold<FormatStringConcatOp>(
        printOp.getLoc(), clonedOperands);
  else
    procPrintInput = clonedOperands.front();

  // Create the procedural print operation and prune the operations outside of
  // the TriggeredOp.
  builder.create<PrintFormattedProcOp>(procPrintInput);
  cleanupList.push_back(printOp.getInput().getDefiningOp());
  printOp.erase();

  builder.create<YieldSeqOp>();

  return success();
}

LogicalResult ProceduralizeSimPass::proceduralize(DPICallOp callOp) {
  assert(!!callOp.getTrigger() && "Cannot proceduralize unclocked dpi call");
  ImplicitLocOpBuilder builder(callOp.getLoc(), callOp);

  auto procOp = builder.create<sim::TriggeredOp>(
      callOp.getResultTypes(), callOp.getTrigger(), callOp.getEnable(),
      callOp.getInputs(), callOp.getTieoffsAttr());
  auto body = builder.createBlock(&procOp.getBody());
  builder.setInsertionPointToStart(body);
  SmallVector<Location> locs(callOp.getInputs().size(), callOp.getLoc());
  procOp.getBody().addArguments(callOp.getInputs().getTypes(), locs);

  auto funcOp = builder.create<ProcCallOp>(
      callOp.getResultTypes(), callOp.getCallee(), true, body->getArguments());
  SmallVector<Value> results(funcOp.getResults().begin(),
                             funcOp.getResults().end());
  builder.create<YieldSeqOp>(results);
  callOp.replaceAllUsesWith(procOp.getResults());
  callOp.erase();
  return success();
}

void ProceduralizeSimPass::proceduralize(FinishOp finishOp) {
  ImplicitLocOpBuilder builder(finishOp.getLoc(), finishOp);

  auto procOp = builder.create<sim::TriggeredOp>(
      TypeRange{}, finishOp.getTrig(), finishOp.getCond(), ValueRange{},
      ArrayAttr{});
  auto body = builder.createBlock(&procOp.getBody());
  builder.setInsertionPointToStart(body);

  builder.create<FinishProcOp>();
  builder.create<YieldSeqOp>();
  finishOp.erase();
}

// Prune the DAGs of formatting tokens left outside of the newly created
// TriggeredOps.
void ProceduralizeSimPass::cleanup() {
  SmallVector<Operation *> cleanupNextList;
  SmallDenseSet<Operation *> erasedOps;

  bool noChange = true;
  while (!cleanupList.empty() || !cleanupNextList.empty()) {

    if (cleanupList.empty()) {
      if (noChange)
        break;
      cleanupList = std::move(cleanupNextList);
      cleanupNextList = {};
      noChange = true;
    }

    auto *opToErase = cleanupList.pop_back_val();
    if (erasedOps.contains(opToErase))
      continue;

    if (opToErase->getUses().empty()) {
      // Remove a dead op. If it is a concat remove its operands, too.
      if (auto concat = dyn_cast<FormatStringConcatOp>(opToErase))
        for (auto operand : concat.getInputs())
          cleanupNextList.push_back(operand.getDefiningOp());
      opToErase->erase();
      erasedOps.insert(opToErase);
      noChange = false;
    } else {
      // Op still has uses, revisit later.
      cleanupNextList.push_back(opToErase);
    }
  }
}

void ProceduralizeSimPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n");
  cleanupList.clear();

  SmallVector<PrintFormattedOp> printOps;
  SmallVector<DPICallOp> dpiCallOps;

  auto theModule = getOperation();
  // Collect printf operations grouped by their clock.
  theModule.walk([&](Operation *op) {
    if (auto printOp = dyn_cast<PrintFormattedOp>(op))
      printOps.push_back(printOp);
    else if (auto dpiCallOp = dyn_cast<DPICallOp>(op))
      if (!!dpiCallOp.getTrigger())
        dpiCallOps.push_back(dpiCallOp);
    if (auto finishOp = dyn_cast<FinishOp>(op))
      proceduralize(finishOp);
  });

  for (auto printOp : printOps) {
    if (failed(proceduralize(printOp))) {
      signalPassFailure();
      return;
    }
  }

  for (auto dpiCallOp : dpiCallOps) {
    if (failed(proceduralize(dpiCallOp))) {
      signalPassFailure();
      return;
    }
  }

  cleanup();
}
