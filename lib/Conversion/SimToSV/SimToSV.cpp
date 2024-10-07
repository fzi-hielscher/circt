//===- LowerSimToSV.cpp - Sim to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translates Sim ops to SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SimToSV.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "lower-sim-to-sv"

namespace circt {
#define GEN_PASS_DEF_LOWERSIMTOSV
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace sim;

namespace {

struct SimConversionState {
  hw::HWModuleOp module;
  bool usedSynthesisMacro = false;
  SetVector<StringAttr> dpiCallees;
};

template <typename T>
struct SimConversionPattern : public OpConversionPattern<T> {
  explicit SimConversionPattern(MLIRContext *context, SimConversionState &state)
      : OpConversionPattern<T>(context), state(state) {}

  SimConversionState &state;
};

} // namespace

// Lower `sim.plusargs.test` to a standard SV implementation.
//
class PlusArgsTestLowering : public SimConversionPattern<PlusArgsTestOp> {
public:
  using SimConversionPattern<PlusArgsTestOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(PlusArgsTestOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto resultType = rewriter.getIntegerType(1);
    auto str = rewriter.create<sv::ConstantStrOp>(loc, op.getFormatString());
    auto reg = rewriter.create<sv::RegOp>(loc, resultType,
                                          rewriter.getStringAttr("_pargs"));
    rewriter.create<sv::InitialOp>(loc, [&] {
      auto call = rewriter.create<sv::SystemFunctionOp>(
          loc, resultType, "test$plusargs", ArrayRef<Value>{str});
      rewriter.create<sv::BPAssignOp>(loc, reg, call);
    });

    rewriter.replaceOpWithNewOp<sv::ReadInOutOp>(op, reg);
    return success();
  }
};

// Lower `sim.plusargs.value` to a standard SV implementation.
//
class PlusArgsValueLowering : public SimConversionPattern<PlusArgsValueOp> {
public:
  using SimConversionPattern<PlusArgsValueOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(PlusArgsValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    auto i1ty = rewriter.getIntegerType(1);
    auto type = op.getResult().getType();

    auto regv = rewriter.create<sv::RegOp>(loc, type,
                                           rewriter.getStringAttr("_pargs_v_"));
    auto regf = rewriter.create<sv::RegOp>(loc, i1ty,
                                           rewriter.getStringAttr("_pargs_f"));

    state.usedSynthesisMacro = true;
    rewriter.create<sv::IfDefOp>(
        loc, "SYNTHESIS",
        [&]() {
          auto cstFalse = rewriter.create<hw::ConstantOp>(loc, APInt(1, 0));
          auto cstZ = rewriter.create<sv::ConstantZOp>(loc, type);
          auto assignZ = rewriter.create<sv::AssignOp>(loc, regv, cstZ);
          circt::sv::setSVAttributes(
              assignZ,
              sv::SVAttributeAttr::get(
                  rewriter.getContext(),
                  "This dummy assignment exists to avoid undriven lint "
                  "warnings (e.g., Verilator UNDRIVEN).",
                  /*emitAsComment=*/true));
          rewriter.create<sv::AssignOp>(loc, regf, cstFalse);
        },
        [&]() {
          rewriter.create<sv::InitialOp>(loc, [&] {
            auto zero32 = rewriter.create<hw::ConstantOp>(loc, APInt(32, 0));
            auto tmpResultType = rewriter.getIntegerType(32);
            auto str =
                rewriter.create<sv::ConstantStrOp>(loc, op.getFormatString());
            auto call = rewriter.create<sv::SystemFunctionOp>(
                loc, tmpResultType, "value$plusargs",
                ArrayRef<Value>{str, regv});
            auto test = rewriter.create<comb::ICmpOp>(
                loc, comb::ICmpPredicate::ne, call, zero32, true);
            rewriter.create<sv::BPAssignOp>(loc, regf, test);
          });
        });

    auto readf = rewriter.create<sv::ReadInOutOp>(loc, regf);
    auto readv = rewriter.create<sv::ReadInOutOp>(loc, regv);
    rewriter.replaceOp(op, {readf, readv});
    return success();
  }
};

static sv::EventControl convertEventControl(hw::EventControl eventCtrl) {
  switch (eventCtrl) {
  case hw::EventControl::AtEdge:
    return sv::EventControl::AtEdge;
  case hw::EventControl::AtPosEdge:
    return sv::EventControl::AtPosEdge;
  case hw::EventControl::AtNegEdge:
    return sv::EventControl::AtNegEdge;
  }
  assert(false && "Invalid event control attr");
}

template <typename FromOp, typename ToOp>
class SimulatorStopLowering : public SimConversionPattern<FromOp> {
public:
  using SimConversionPattern<FromOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(FromOp op, typename FromOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    Value clockCast = rewriter.create<seq::FromClockOp>(loc, adaptor.getClk());

    this->state.usedSynthesisMacro = true;
    rewriter.create<sv::IfDefOp>(
        loc, "SYNTHESIS", [&] {},
        [&] {
          rewriter.create<sv::AlwaysOp>(
              loc, sv::EventControl::AtPosEdge, clockCast, [&] {
                rewriter.create<sv::IfOp>(loc, adaptor.getCond(),
                                          [&] { rewriter.create<ToOp>(loc); });
              });
        });

    rewriter.eraseOp(op);

    return success();
  }
};

class DPICallLowering : public SimConversionPattern<DPICallOp> {
public:
  using SimConversionPattern<DPICallOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(DPICallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    // Record the callee.
    state.dpiCallees.insert(op.getCalleeAttr().getAttr());

    bool isClockedCall = !!op.getTrigger();

    bool hasEnable = !!op.getEnable();

    SmallVector<sv::RegOp> temporaries;
    SmallVector<Value> reads;
    for (auto [type, result] :
         llvm::zip(op.getResultTypes(), op.getResults())) {
      temporaries.push_back(rewriter.create<sv::RegOp>(op.getLoc(), type));
      reads.push_back(
          rewriter.create<sv::ReadInOutOp>(op.getLoc(), temporaries.back()));
    }

    auto emitCall = [&]() {
      auto call = rewriter.create<sv::FuncCallProceduralOp>(
          op.getLoc(), op.getResultTypes(), op.getCalleeAttr(),
          adaptor.getInputs());
      for (auto [lhs, rhs] : llvm::zip(temporaries, call.getResults())) {
        if (isClockedCall)
          rewriter.create<sv::PAssignOp>(op.getLoc(), lhs, rhs);
        else
          rewriter.create<sv::BPAssignOp>(op.getLoc(), lhs, rhs);
      }
    };
    if (isClockedCall) {
      Value clockCast = rewriter.create<seq::FromClockOp>(
          loc, sim::getLocalRootTrigger(adaptor.getTrigger())
                   .getDefiningOp<OnEdgeOp>()
                   .getClock());
      rewriter.create<sv::AlwaysOp>(
          loc, ArrayRef<sv::EventControl>{sv::EventControl::AtPosEdge},
          ArrayRef<Value>{clockCast}, [&]() {
            if (!hasEnable)
              return emitCall();
            rewriter.create<sv::IfOp>(op.getLoc(), adaptor.getEnable(),
                                      emitCall);
          });
    } else {
      // Unclocked call is lowered into always_comb.
      // TODO: If there is a return value and no output argument, use an
      // unclocked call op.
      rewriter.create<sv::AlwaysCombOp>(loc, [&]() {
        if (!hasEnable)
          return emitCall();
        auto assignXToResults = [&] {
          for (auto lhs : temporaries) {
            auto xValue = rewriter.create<sv::ConstantXOp>(
                op.getLoc(), lhs.getType().getElementType());
            rewriter.create<sv::BPAssignOp>(op.getLoc(), lhs, xValue);
          }
        };
        rewriter.create<sv::IfOp>(op.getLoc(), adaptor.getEnable(), emitCall,
                                  assignXToResults);
      });
    }

    rewriter.replaceOp(op, reads);
    return success();
  }
};

class ProcCallLowering : public SimConversionPattern<sim::ProcCallOp> {
public:
  using SimConversionPattern<ProcCallOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(ProcCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Record the callee.
    if (op.getDpi())
      state.dpiCallees.insert(op.getCalleeAttr().getAttr());

    rewriter.replaceOpWithNewOp<sv::FuncCallProceduralOp>(
        op, op.getResultTypes(), op.getCalleeAttr(), adaptor.getInputs());

    return success();
  }
};

class FinishProcLowering : public SimConversionPattern<sim::FinishProcOp> {
public:
  using SimConversionPattern<FinishProcOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(FinishProcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<sv::FinishOp>(op);
    return success();
  }
};

// A helper struct to lower DPI function/call.
struct LowerDPIFunc {
  llvm::DenseMap<StringAttr, StringAttr> symbolToFragment;
  circt::Namespace nameSpace;
  LowerDPIFunc(mlir::ModuleOp module) { nameSpace.add(module); }
  void lower(sim::DPIFuncOp func);
  void addFragments(hw::HWModuleOp module,
                    ArrayRef<StringAttr> dpiCallees) const;
};

void LowerDPIFunc::lower(sim::DPIFuncOp func) {
  ImplicitLocOpBuilder builder(func.getLoc(), func);
  ArrayAttr inputLocsAttr, outputLocsAttr;
  if (func.getArgumentLocs()) {
    SmallVector<Attribute> inputLocs, outputLocs;
    for (auto [port, loc] :
         llvm::zip(func.getModuleType().getPorts(),
                   func.getArgumentLocsAttr().getAsRange<LocationAttr>())) {
      (port.dir == hw::ModulePort::Output ? outputLocs : inputLocs)
          .push_back(loc);
    }
    inputLocsAttr = builder.getArrayAttr(inputLocs);
    outputLocsAttr = builder.getArrayAttr(outputLocs);
  }

  auto svFuncDecl =
      builder.create<sv::FuncOp>(func.getSymNameAttr(), func.getModuleType(),
                                 func.getPerArgumentAttrsAttr(), inputLocsAttr,
                                 outputLocsAttr, func.getVerilogNameAttr());
  // DPI function is a declaration so it must be a private function.
  svFuncDecl.setPrivate();
  auto name = builder.getStringAttr(nameSpace.newName(
      func.getSymNameAttr().getValue(), "dpi_import_fragument"));

  // Add include guards to avoid duplicate declarations. See Issue 7458.
  auto macroDecl = builder.create<sv::MacroDeclOp>(nameSpace.newName(
      "__CIRCT_DPI_IMPORT", func.getSymNameAttr().getValue().upper()));
  builder.create<emit::FragmentOp>(name, [&]() {
    builder.create<sv::IfDefOp>(
        macroDecl.getSymNameAttr(), []() {},
        [&]() {
          builder.create<sv::FuncDPIImportOp>(func.getSymNameAttr(),
                                              StringAttr());
          builder.create<sv::MacroDefOp>(macroDecl.getSymNameAttr(), "");
        });
  });

  symbolToFragment.insert({func.getSymNameAttr(), name});
  func.erase();
}

void LowerDPIFunc::addFragments(hw::HWModuleOp module,
                                ArrayRef<StringAttr> dpiCallees) const {
  llvm::SetVector<Attribute> fragments;
  // Add existing emit fragments.
  if (auto exstingFragments =
          module->getAttrOfType<ArrayAttr>(emit::getFragmentsAttrName()))
    for (auto fragment : exstingFragments.getAsRange<FlatSymbolRefAttr>())
      fragments.insert(fragment);
  for (auto callee : dpiCallees) {
    auto attr = symbolToFragment.at(callee);
    fragments.insert(FlatSymbolRefAttr::get(attr));
  }
  if (!fragments.empty())
    module->setAttr(
        emit::getFragmentsAttrName(),
        ArrayAttr::get(module.getContext(), fragments.takeVector()));
}

static Value materializeInitValue(OpBuilder &builder, TypedAttr cst,
                                  Location loc) {
  if (llvm::isa<sv::SVDialect>(cst.getDialect()))
    return cst.getDialect()
        .materializeConstant(builder, cst, cst.getType(), loc)
        ->getResult(0);
  auto *hwDialect = builder.getContext()->getLoadedDialect<hw::HWDialect>();
  return hwDialect->materializeConstant(builder, cst, cst.getType(), loc)
      ->getResult(0);
}

static void cleanUpTriggerTree(ArrayRef<TriggeredOp> procs) {
  SmallVector<Operation *> cleanupList;
  SmallVector<Operation *> cleanupNextList;
  SmallPtrSet<Operation *, 8> erasedOps;
  for (auto proc : procs) {
    auto trigger = proc.getTrigger();
    cleanupNextList.emplace_back(trigger.getDefiningOp());
    erasedOps.insert(proc);
    proc.erase();
  }

  bool hasChanged = true;
  while (hasChanged && !cleanupNextList.empty()) {
    cleanupList = std::move(cleanupNextList);
    cleanupNextList.clear();
    hasChanged = false;

    for (auto op : cleanupList) {
      if (!op || erasedOps.contains(op))
        continue;

      if (auto seqOp = dyn_cast<TriggerSequenceOp>(op)) {
        if (seqOp.use_empty()) {
          cleanupNextList.push_back(seqOp.getParent().getDefiningOp());
          erasedOps.insert(seqOp);
          seqOp.erase();
          hasChanged = true;
        } else {
          cleanupNextList.push_back(seqOp);
        }
        continue;
      }

      if (isa<OnEdgeOp, OnInitOp>(op)) {
        if (op->use_empty()) {
          erasedOps.insert(op);
          op->erase();
        } else {
          cleanupNextList.push_back(op);
        }
        continue;
      }
    }
  }
}

static void cleanUpFormatStringTree(ArrayRef<PrintFormattedProcOp> deadFmts) {
  SmallVector<Operation *> cleanupList;
  SmallVector<Operation *> cleanupNextList;
  SmallPtrSet<Operation *, 8> erasedOps;

  for (auto deadFmt : deadFmts) {
    cleanupNextList.push_back(deadFmt.getInput().getDefiningOp());
    erasedOps.insert(deadFmt);
    deadFmt.erase();
  }

  bool hasChanged = true;
  while (hasChanged && !cleanupNextList.empty()) {
    cleanupList = std::move(cleanupNextList);
    cleanupNextList.clear();
    hasChanged = false;

    for (auto op : cleanupList) {
      if (!op || erasedOps.contains(op))
        continue;

      if (auto concat = dyn_cast<FormatStringConcatOp>(op)) {
        if (!concat->use_empty()) {
          cleanupNextList.push_back(concat);
          continue;
        }
        for (auto arg : concat.getInputs())
          cleanupNextList.emplace_back(arg.getDefiningOp());
        hasChanged = true;
        erasedOps.insert(concat);
        concat.erase();
        continue;
      }

      if (isa<FormatBinOp, FormatHexOp, FormatDecOp, FormatCharOp, FormatLitOp>(
              op)) {
        if (op->use_empty()) {
          erasedOps.insert(op);
          op->erase();
        } else {
          cleanupNextList.push_back(op);
        }
        continue;
      }
    }
  }
}

static LogicalResult lowerRootTrigger(Value root, ArrayRef<TriggeredOp> procs) {
  auto rootDefOp = root.getDefiningOp();
  assert(!!rootDefOp);
  OpBuilder builder(rootDefOp);
  SmallVector<Location> locs;
  locs.reserve(procs.size() + 1);
  locs.emplace_back(rootDefOp->getLoc());

  SmallDenseMap<TriggeredOp, SmallVector<Value>> resultToRegMap;

  for (auto proc : procs) {
    locs.emplace_back(proc.getLoc());
    if (proc.getNumResults() > 0) {
      SmallVector<Value> newRegs;
      SmallVector<Value> reads;
      for (auto [res, tieoff] :
           llvm::zip(proc.getResults(), *proc.getTieoffs())) {
        auto cst = materializeInitValue(builder, cast<TypedAttr>(tieoff),
                                        proc.getLoc());
        auto reg =
            builder.create<sv::RegOp>(proc.getLoc(), res.getType(),
                                      StringAttr(), hw::InnerSymAttr(), cst);
        auto regRead = builder.createOrFold<sv::ReadInOutOp>(proc.getLoc(),
                                                             reg.getResult());
        newRegs.emplace_back(reg.getResult());
        reads.emplace_back(regRead);
      }
      proc.replaceAllUsesWith(reads);
      resultToRegMap.insert(
          std::pair<TriggeredOp, SmallVector<Value>>{proc, std::move(newRegs)});
    }
  }

  auto fusedLoc = FusedLoc::get(builder.getContext(), locs);

  struct BuildStackEntry {
    PointerUnion<Value, Operation *> pv;
    OpBuilder::InsertPoint ip;
  };

  auto ifDefOp =
      builder.create<sv::IfDefOp>(fusedLoc, "SYNTHESIS", [] {}, [] {});
  builder.setInsertionPointToStart(ifDefOp.getElseBlock());

  if (isa<OnInitOp>(rootDefOp)) {
    auto initOp = builder.create<sv::InitialOp>(fusedLoc);
    builder.setInsertionPointToStart(initOp.getBodyBlock());
  } else if (auto edgeOp = dyn_cast<OnEdgeOp>(rootDefOp)) {
    SmallVector<Value, 1> convClock;
    builder.createOrFold<mlir::UnrealizedConversionCastOp>(
        convClock, fusedLoc, TypeRange{builder.getI1Type()}, edgeOp.getClock());
    auto alwaysOp = builder.create<sv::AlwaysFFOp>(
        fusedLoc, convertEventControl(edgeOp.getEvent()), convClock.front());
    builder.setInsertionPointToStart(alwaysOp.getBodyBlock());
  } else {
    root.getDefiningOp()->emitError("Unsupported trigger root.");
    return failure();
  }

  SmallVector<BuildStackEntry> buildStack;
  buildStack.emplace_back(BuildStackEntry{root, builder.saveInsertionPoint()});
  while (!buildStack.empty()) {
    auto popVal = buildStack.pop_back_val();
    builder.restoreInsertionPoint(popVal.ip);

    if (auto trigVal = dyn_cast<Value>(popVal.pv)) {
      auto users = trigVal.getUsers();
      if (users.empty())
        continue;
      if (trigVal.hasOneUse()) {
        buildStack.emplace_back(
            BuildStackEntry{users.begin().getCurrent().getUser(), popVal.ip});
        continue;
      }
      SmallVector<Operation *> userOps(trigVal.getUsers().begin(),
                                       trigVal.getUsers().end());
      auto forkJoinOp =
          builder.create<sv::ForkJoinOp>(rootDefOp->getLoc(), userOps.size());
      for (auto const &[user, region] :
           llvm::reverse(llvm::zip(userOps, forkJoinOp.getRegions()))) {
        Block *block = builder.createBlock(&region);
        auto newIp = OpBuilder::InsertPoint(block, block->begin());
        buildStack.emplace_back(BuildStackEntry{user, newIp});
      }
      continue;
    }

    auto op = cast<Operation *>(popVal.pv);

    if (auto sequence = dyn_cast<TriggerSequenceOp>(op)) {
      for (auto res : llvm::reverse(sequence.getResults()))
        buildStack.emplace_back(BuildStackEntry{res, popVal.ip});
      continue;
    }

    auto procedure = dyn_cast<TriggeredOp>(op);
    if (!procedure) {
      op->emitWarning("Unable to lower trigger user.");
      continue;
    }

    auto enable = procedure.getCondition();
    bool condional = false;
    if (enable) {
      if (auto constEn = enable.getDefiningOp<hw::ConstantOp>()) {
        if (constEn.getValue().isZero())
          continue;
      } else {
        condional = true;
      }
    }

    auto yield = cast<YieldSeqOp>(procedure.getBody().front().getTerminator());
    auto &regs = resultToRegMap[procedure];
    mlir::IRRewriter rewriter(builder);

    if (condional) {
      auto ifOp = builder.create<sv::IfOp>(procedure.getLoc(), enable);
      builder.setInsertionPointToStart(ifOp.getThenBlock());
    }

    OpBuilder::InsertPoint ip = builder.saveInsertionPoint();
    rewriter.inlineBlockBefore(&procedure.getBody().front(), ip.getBlock(),
                               ip.getPoint(), procedure.getInputs());

    assert(regs.size() == yield.getNumOperands() &&
           "Failed to lookup materialized result registers");
    for (auto [reg, res] : llvm::zip(regs, yield.getOperands()))
      builder.create<sv::PAssignOp>(yield.getLoc(), reg, res);
    yield.erase();
  }

  return success();
}

static LogicalResult lowerProcPrint(OpBuilder &builder,
                                    PrintFormattedProcOp printOp) {
  SmallVector<Value, 4> flatString;
  if (auto concat = printOp.getInput().getDefiningOp<FormatStringConcatOp>()) {
    auto isAcyclic = concat.getFlattenedInputs(flatString);
    if (failed(isAcyclic))
      return printOp.emitOpError("Format string is cyclic.");
  } else {
    flatString.push_back(printOp.getInput());
  }

  SmallString<64> fmtString;
  SmallVector<Value> substitutions;
  SmallVector<Location> locs;
  for (auto fmt : flatString) {
    auto defOp = fmt.getDefiningOp();
    if (!defOp)
      return printOp.emitError(
          "Formatting tokens must not be passed as arguments.");
    bool ok =
        llvm::TypeSwitch<Operation *, bool>(defOp)
            .Case<FormatLitOp>([&](auto literal) {
              fmtString.reserve(fmtString.size() + literal.getLiteral().size());
              for (auto c : literal.getLiteral()) {
                fmtString.push_back(c);
                if (c == '%')
                  fmtString.push_back('%');
              }
              return true;
            })
            .Case<FormatBinOp>([&](auto bin) {
              fmtString.push_back('%');
              fmtString.push_back('b');
              substitutions.push_back(bin.getValue());
              return true;
            })
            .Case<FormatDecOp>([&](auto dec) {
              fmtString.push_back('%');
              fmtString.push_back('d');
              Type ty = dec.getValue().getType();
              Value conv = builder.createOrFold<sv::SystemFunctionOp>(
                  dec.getLoc(), ty, dec.getIsSigned() ? "signed" : "unsigned",
                  dec.getValue());
              substitutions.push_back(conv);
              return true;
            })
            .Case<FormatHexOp>([&](auto hex) {
              fmtString.push_back('%');
              fmtString.push_back('h');
              substitutions.push_back(hex.getValue());
              return true;
            })
            .Case<FormatCharOp>([&](auto c) {
              fmtString.push_back('%');
              fmtString.push_back('c');
              substitutions.push_back(c.getValue());
              return true;
            })
            .Default([&](Operation *op) {
              op->emitError("Unsupported format specifier op.");
              return false;
            });
    if (!ok)
      return failure();
    locs.push_back(defOp->getLoc());
  }
  locs.push_back(printOp.getLoc());
  auto fusedLoc = FusedLoc::get(builder.getContext(), locs);
  Value stdErr = builder.createOrFold<hw::ConstantOp>(
      printOp.getLoc(), builder.getI32IntegerAttr(0x80000002));
  builder.create<sv::FWriteOp>(fusedLoc, stdErr,
                               builder.getStringAttr(fmtString), substitutions);
  return success();
}

static LogicalResult lowerProceuralRegion(Region &body) {
  SmallVector<PrintFormattedProcOp> printCleanupList;

  bool hasFailed = false;
  body.walk([&](PrintFormattedProcOp printOp) {
    OpBuilder builder(printOp);
    if (failed(lowerProcPrint(builder, printOp)))
      hasFailed = true;
    printCleanupList.push_back(printOp);
  });

  if (hasFailed)
    return failure();

  cleanUpFormatStringTree(printCleanupList);
  return success(!hasFailed);
}

static LogicalResult lowerTriggeredOps(hw::HWModuleOp hwModuleOp) {
  SmallDenseMap<Value, SmallVector<TriggeredOp>, 2> rootTriggers;

  for (auto procOp : hwModuleOp.getOps<TriggeredOp>()) {
    auto localRoot = getLocalRootTrigger(procOp.getTrigger());
    if (!localRoot) {
      procOp.emitError("Unable to find root trigger.");
      return failure();
    }
    if (isa<BlockArgument>(localRoot)) {
      procOp.emitError(
          "Lowering cross-module triggers is currently unsupported.");
      return failure();
    }
    rootTriggers[localRoot].emplace_back(procOp);
    if (failed(lowerProceuralRegion(procOp.getBody())))
      return failure();
  }

  OpBuilder builder(hwModuleOp);
  for (auto &[root, procs] : rootTriggers)
    if (succeeded(lowerRootTrigger(root, procs))) {
      cleanUpTriggerTree(procs);
    } else {
      return failure();
    }

  return success();
}

namespace {
struct SimToSVPass : public circt::impl::LowerSimToSVBase<SimToSVPass> {
  void runOnOperation() override {
    auto circuit = getOperation();
    MLIRContext *context = &getContext();
    LowerDPIFunc lowerDPIFunc(circuit);

    // Lower DPI functions.
    for (auto func :
         llvm::make_early_inc_range(circuit.getOps<sim::DPIFuncOp>()))
      lowerDPIFunc.lower(func);

    std::atomic<bool> usedSynthesisMacro = false;
    auto lowerModule = [&](hw::HWModuleOp moduleOp) {
      if (failed(lowerTriggeredOps(moduleOp)))
        return failure();

      SimConversionState state;
      ConversionTarget target(*context);
      target.addIllegalDialect<SimDialect>();
      target.addLegalDialect<sv::SVDialect>();
      target.addLegalDialect<hw::HWDialect>();
      target.addLegalDialect<seq::SeqDialect>();
      target.addLegalDialect<comb::CombDialect>();

      RewritePatternSet patterns(context);
      patterns.add<PlusArgsTestLowering>(context, state);
      patterns.add<PlusArgsValueLowering>(context, state);

      patterns.add<SimulatorStopLowering<sim::FatalOp, sv::FatalOp>>(context,
                                                                     state);
      patterns.add<FinishProcLowering>(context, state);
      patterns.add<DPICallLowering>(context, state);
      patterns.add<ProcCallLowering>(context, state);

      auto result =
          applyPartialConversion(moduleOp, target, std::move(patterns));

      if (failed(result))
        return result;

      // Set the emit fragment.
      lowerDPIFunc.addFragments(moduleOp, state.dpiCallees.takeVector());

      //  if (state.usedSynthesisMacro)
      usedSynthesisMacro = true;
      return result;
    };

    if (failed(mlir::failableParallelForEach(
            context, circuit.getOps<hw::HWModuleOp>(), lowerModule)))
      return signalPassFailure();

    if (usedSynthesisMacro) {
      Operation *op = circuit.lookupSymbol("SYNTHESIS");
      if (op) {
        if (!isa<sv::MacroDeclOp>(op)) {
          op->emitOpError("should be a macro declaration");
          return signalPassFailure();
        }
      } else {
        auto builder = ImplicitLocOpBuilder::atBlockBegin(
            UnknownLoc::get(context), circuit.getBody());
        builder.create<sv::MacroDeclOp>("SYNTHESIS");
      }
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> circt::createLowerSimToSVPass() {
  return std::make_unique<SimToSVPass>();
}
