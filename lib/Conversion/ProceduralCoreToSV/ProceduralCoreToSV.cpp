//===- ProceduralCoreToSV.cpp - Procedural Core To SV Conversion Pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower procedural core dialect (HW/Sim) operations to the SV dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ProceduralCoreToSV.h"

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

namespace circt {
#define GEN_PASS_DEF_PROCEDURALCORETOSV
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace mlir;

static sv::EventControl hwToSvEventControl(hw::EventControl ec) {
  switch (ec) {
  case hw::EventControl::AtPosEdge:
    return sv::EventControl::AtPosEdge;
  case hw::EventControl::AtNegEdge:
    return sv::EventControl::AtNegEdge;
  case hw::EventControl::AtEdge:
    return sv::EventControl::AtEdge;
  }
  llvm_unreachable("Unknown event control kind");
}

namespace {

struct ProceduralOpRewriter : public RewriterBase {
  ProceduralOpRewriter(MLIRContext *ctxt) : RewriterBase::RewriterBase(ctxt) {}
};

struct ProceduralCoreToSVPass
    : public circt::impl::ProceduralCoreToSVBase<ProceduralCoreToSVPass> {
  ProceduralCoreToSVPass() = default;
  void runOnOperation() override;
};
} // namespace

static LogicalResult convertSCFIfToSVIf(scf::IfOp &scfIfOp,
                                        ProceduralOpRewriter &rewriter) {
  // For each SCF If op create a new SV If op, move the nested blocks over
  // and erase the SCF If op.
  if (scfIfOp.getNumResults() != 0) {
    scfIfOp.emitError(
        "SCF If operation with results cannot be converted to SV.");
    return failure();
  }

  rewriter.setInsertionPoint(scfIfOp);
  auto svIfOp =
      rewriter.create<sv::IfOp>(scfIfOp.getLoc(), scfIfOp.getCondition());

  rewriter.eraseOp(scfIfOp.getBody()->getTerminator());
  rewriter.mergeBlocks(scfIfOp.getBody(), svIfOp.getBody());

  if (!scfIfOp.getElseRegion().empty()) {
    auto *elseBlock = &scfIfOp.getElseRegion().front();
    rewriter.eraseOp(elseBlock->getTerminator());
    rewriter.moveBlockBefore(elseBlock, &svIfOp.getElseRegion(),
                             svIfOp.getElseRegion().begin());
  }

  rewriter.eraseOp(scfIfOp);
  return success();
}

static FailureOr<Value> processFormatToken(sim::PrintFormattedProcOp &printOp,
                                           Operation *fmtOp,
                                           ImplicitLocOpBuilder &builder,
                                           SmallString<32> &strBuffer) {
  if (!fmtOp) {
    printOp.emitError(
        "Format strings passed as block argument cannot be lowered.");
    return FailureOr<Value>();
  }

  // Append the format string and get the substitution value, if present.
  return llvm::TypeSwitch<Operation *, FailureOr<Value>>(fmtOp)
      .Case<sim::FormatLitOp>([&](sim::FormatLitOp litOp) -> FailureOr<Value> {
        // Copy escaped literal.
        for (char c : litOp.getLiteral()) {
          strBuffer.push_back(c);
          if (c == '%')
            strBuffer.push_back('%');
        }
        return Value();
      })
      .Case<sim::FormatHexOp>([&](sim::FormatHexOp hexOp) -> FailureOr<Value> {
        strBuffer += "%x";
        return hexOp.getValue();
      })
      .Case<sim::FormatDecOp>([&](sim::FormatDecOp decOp) -> FailureOr<Value> {
        strBuffer += "%d";
        return builder.createOrFold<sv::SystemFunctionOp>(
            decOp.getValue().getType(),
            decOp.getIsSigned() ? "signed" : "unsigned", decOp.getValue());
      })
      .Case<sim::FormatBinOp>([&](sim::FormatBinOp binOp) -> FailureOr<Value> {
        strBuffer += "%b";
        return binOp.getValue();
      })
      .Case<sim::FormatCharOp>(
          [&](sim::FormatCharOp charOp) -> FailureOr<Value> {
            strBuffer += "%c";
            return charOp.getValue();
          })
      .Default([&](Operation *op) -> FailureOr<Value> {
        assert(!llvm::isa<sim::FormatStringConcatOp>(op) &&
               "Concat should have been flattened.");
        op->emitError("Unsupported format string operation.");
        printOp.emitError("Failed to lower format string.");
        return FailureOr<Value>();
      });
}

static LogicalResult lowerProcPrint(sim::PrintFormattedProcOp &printOp,
                                    ProceduralOpRewriter &rewriter, Value fd) {
  rewriter.setInsertionPoint(printOp);

  SmallVector<Value> flatTokens;
  SmallVector<Value> substitutions;
  SmallString<32> svFormatString;

  ImplicitLocOpBuilder builder(printOp.getLoc(), printOp);

  // Get the flat list of format string tokens.
  if (auto concatOp =
          printOp.getInput().getDefiningOp<sim::FormatStringConcatOp>()) {
    auto isAcyclic = concatOp.getFlattenedInputs(flatTokens);
    (void)isAcyclic;
    assert(succeeded(isAcyclic) &&
           "Cyclic concatenation in a procedural region encountered.");
  } else {
    flatTokens.push_back(printOp.getInput());
  }

  // Assemble the SV format string and collect the substitutuon values.
  for (auto token : flatTokens) {
    auto subst = processFormatToken(printOp, token.getDefiningOp(), builder,
                                    svFormatString);
    if (failed(subst))
      return failure();
    if (*subst)
      substitutions.push_back(*subst);
  }

  // Exchange the ops.
  rewriter.createOrFold<sv::FWriteOp>(printOp.getLoc(), fd,
                                      builder.getStringAttr(svFormatString),
                                      substitutions);

  rewriter.eraseOp(printOp);
  return success();
}

static LogicalResult lowerAlwaysOpBody(sv::AlwaysOp alwaysOp) {
  ProceduralOpRewriter bodyRewriter(alwaysOp.getContext());
  SmallVector<Operation *> fstringOps;

  // Create a constant for the file descriptor to print to.
  // For compatibility with previous FIRRTL lowerings we print
  // to stderr (0x80000002).
  bodyRewriter.setInsertionPointToStart(alwaysOp.getBodyBlock());
  hw::ConstantOp outputFDCst = bodyRewriter.create<hw::ConstantOp>(
      alwaysOp.getLoc(), APInt(32, 0x80000002));

  bool fdCstUsed = false;
  bool conversionFailed = false;

  // Walk the body, perform conversions as we go.
  alwaysOp.walk<WalkOrder::PostOrder>([&](Operation *op) -> WalkResult {
    if (auto scfIfOp = llvm::dyn_cast<scf::IfOp>(op)) {
      // Convert SCF If to SV If.
      if (failed(convertSCFIfToSVIf(scfIfOp, bodyRewriter)))
        conversionFailed = true;
    } else if (auto printOp = llvm::dyn_cast<sim::PrintFormattedProcOp>(op)) {
      // Convert sim procedural print to SystemVerilog fwrite.
      if (failed(
              lowerProcPrint(printOp, bodyRewriter, outputFDCst.getResult())))
        conversionFailed = true;
      fdCstUsed = true;
    } else if (sim::isFormatStringOperation(op)) {
      // Collect format string token ops for cleanup.
      fstringOps.push_back(op);
    }
    return WalkResult::advance();
  });

  if (conversionFailed)
    return failure();

  // Tidy up format string tokens
  while (!fstringOps.empty()) {
    size_t oldSize = fstringOps.size();
    SmallVector<Operation *> cleanupList = std::move(fstringOps);
    fstringOps = SmallVector<Operation *>();
    for (auto tokenOp : cleanupList) {
      assert(tokenOp->getNumResults() == 1 && "Unexpected operation.");
      if (tokenOp->getResult(0).getUses().empty())
        tokenOp->erase();
      else
        fstringOps.push_back(tokenOp);
    }
    if (fstringOps.size() == oldSize)
      break;
  }

  // Remove the file descriptor constant if we never used it.
  if (!fdCstUsed)
    bodyRewriter.eraseOp(outputFDCst);

  return success();
}

void ProceduralCoreToSVPass::runOnOperation() {
  hw::HWModuleOp theModule = getOperation();

  SmallVector<sv::AlwaysOp> alwaysOps;
  ProceduralOpRewriter moduleRewriter(theModule.getContext());

  theModule.walk<WalkOrder::PreOrder>([&](hw::TriggeredOp triggeredOp)
                                          -> WalkResult {
    // Create an AlwaysOp, move the body over and remove the TriggeredOp
    moduleRewriter.setInsertionPoint(triggeredOp);
    auto alwaysOp = moduleRewriter.create<sv::AlwaysOp>(
        triggeredOp.getLoc(),
        ArrayRef<sv::EventControl>{hwToSvEventControl(triggeredOp.getEvent())},
        ArrayRef<Value>{triggeredOp.getTrigger()});
    moduleRewriter.mergeBlocks(triggeredOp.getBodyBlock(),
                               alwaysOp.getBodyBlock(),
                               triggeredOp.getInputs());
    moduleRewriter.eraseOp(triggeredOp);
    alwaysOps.push_back(alwaysOp);
    // Don't recurse into the body.
    return WalkResult::skip();
  });

  // Lower the body region
  auto passResult = failableParallelForEach(theModule.getContext(), alwaysOps,
                                            lowerAlwaysOpBody);
  if (failed(passResult))
    signalPassFailure();
}
