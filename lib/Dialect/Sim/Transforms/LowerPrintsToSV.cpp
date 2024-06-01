#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "circt/Support/Debug.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/Pass.h"
#include <string.h>

#define DEBUG_TYPE "sim-lower-prints-to-sv"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_LOWERPRINTSTOSV
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace llvm;
using namespace circt;
using namespace sim;

namespace {
struct LowerPrintsToSVPass : impl::LowerPrintsToSVBase<LowerPrintsToSVPass> {
public:
  void runOnOperation() override;

private:
};
} // namespace

struct PrintOpRewriter : public OpRewritePattern<sim::PrintFormattedProcOp> {
public:
  using OpRewritePattern<PrintFormattedProcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PrintFormattedProcOp op,
                                PatternRewriter &rewriter) const final;

private:
  FailureOr<Value> processToken(Operation *fmtOp, PatternRewriter &rewriter,
                                SmallString<32> &strBuffer) const;
};

// Hack: This rewrite does not really belong here, but where else to put it?
struct SCFIfOpRewriter : public OpRewritePattern<mlir::scf::IfOp> {
public:
  using OpRewritePattern<mlir::scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::IfOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op.thenYield());
    auto ifOp = rewriter.create<sv::IfOp>(op.getLoc(), op.getCondition());
    rewriter.moveBlockBefore(op.thenBlock(), ifOp.getThenBlock());
    rewriter.eraseBlock(&*(++(ifOp.getThenRegion().getBlocks().begin())));
    rewriter.eraseOp(op);
    return success();
  }
};

FailureOr<Value>
PrintOpRewriter::processToken(Operation *fmtOp, PatternRewriter &rewriter,
                              SmallString<32> &strBuffer) const {
  if (!fmtOp)
    return FailureOr<Value>();

  return llvm::TypeSwitch<Operation *, FailureOr<Value>>(fmtOp)
      .Case<sim::FormatLitOp>([&](sim::FormatLitOp litOp) -> FailureOr<Value> {
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
        return decOp.getValue();
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
        op->emitWarning("Unsupported format operation.");
        return FailureOr<Value>();
      });
}

LogicalResult
PrintOpRewriter::matchAndRewrite(PrintFormattedProcOp op,
                                 PatternRewriter &rewriter) const {

  SmallVector<Value> substitutions;
  SmallString<32> fmtStr;

  for (auto operand : op.getInputs()) {
    auto subst = processToken(operand.getDefiningOp(), rewriter, fmtStr);
    if (failed(subst))
      return op.emitError("Unable to create substitutions for format string");

    if (!(*subst))
      continue;

    if (auto decFmt = operand.getDefiningOp<sim::FormatDecOp>()) {
      if (decFmt.getIsSigned()) {
        auto signedWrap = rewriter.createOrFold<sv::SystemFunctionOp>(
            op.getLoc(), subst->getType(), "signed", *subst);
        substitutions.push_back(signedWrap);
        continue;
      }
    }

    substitutions.push_back(*subst);
  }

  auto fdErr =
      rewriter.createOrFold<hw::ConstantOp>(op.getLoc(), APInt(32, 0x80000002));
  rewriter.replaceOpWithNewOp<sv::FWriteOp>(
      op, fdErr, rewriter.getStringAttr(fmtStr), substitutions);
  return success();
}

void LowerPrintsToSVPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n");

  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = false;
  config.maxIterations = 2;

  RewritePatternSet convPatterns(&getContext());
  convPatterns.add<PrintOpRewriter>(&getContext());
  convPatterns.add<SCFIfOpRewriter>(&getContext());

  if (failed(mlir::applyPatternsAndFoldGreedily(
          getOperation(), std::move(convPatterns), config))) {
    signalPassFailure();
    return;
  }
}
