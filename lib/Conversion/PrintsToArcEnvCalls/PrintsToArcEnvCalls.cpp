//===- LowerArcToLLVM.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-prints-to-arc-env-calls"

namespace circt {
#define GEN_PASS_DEF_LOWERPRINTSTOARCENVCALLS
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

static LLVM::LLVMFuncOp
lookupOrInsertExternalFunction(OpBuilder &builder, mlir::ModuleOp moduleOp,
                               StringRef symbolName,
                               LLVM::LLVMFunctionType fnType) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto func = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(symbolName);
  if (func)
    return func;
  func = builder.create<LLVM::LLVMFuncOp>(moduleOp.getLoc(), symbolName, fnType,
                                          LLVM::Linkage::External,
                                          /*dsoLocal=*/false, LLVM::CConv::C);
  return func;
}

namespace {

struct ArcEnvCallInfo {
  LLVM::LLVMFuncOp getPrintStreamFunc = {};
  LLVM::LLVMFuncOp printfFunc = {};
  LLVM::LLVMFuncOp putsFunc = {};
  LLVM::LLVMFuncOp putcFunc = {};
  bool printfUsed = false;
  bool putsUsed = false;
  bool putcUsed = false;
};

struct LowerPrintsToArcEnvCallsPass
    : public circt::impl::LowerPrintsToArcEnvCallsBase<
          LowerPrintsToArcEnvCallsPass> {

  using circt::impl::LowerPrintsToArcEnvCallsBase<
      LowerPrintsToArcEnvCallsPass>::LowerPrintsToArcEnvCallsBase;

  LogicalResult convertPrintOp(RewriterBase &rewriter,
                               sim::PrintFormattedProcOp printOp);

  LogicalResult lowerPrints(RewriterBase &rewriter,
                            ArrayRef<sim::PrintFormattedProcOp> printOps);

  LLVM::GlobalOp lookupOrCreateStringSymbol(RewriterBase &rewriter,
                                            StringAttr str);

  SmallVector<Value> cleanupList;
  ArcEnvCallInfo envCallInfo;
  SmallDenseMap<StringAttr, LLVM::GlobalOp> stringSymbolCache;

  void runOnOperation() override;
};

struct FormatDecOpLowering : public OpConversionPattern<sim::FormatDecOp> {

  using OpConversionPattern::OpConversionPattern;

  // Create the decimal string representation of an arbitray integer
  LogicalResult
  matchAndRewrite(sim::FormatDecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Type = rewriter.getI32Type();
    auto i8Type = rewriter.getI8Type();

    unsigned width = adaptor.getValue().getType().getIntOrFloatBitWidth();

    // Allocate the reqired number of characters (including zero terminator) on
    // the stack.
    auto bufBytes =
        sim::FormatDecOp::getDecimalWidth(width, op.getIsSigned()) + 1;
    assert(bufBytes > 1);
    auto bufBytesCst =
        rewriter.createOrFold<LLVM::ConstantOp>(op.getLoc(), i32Type, bufBytes);
    auto allocOp = rewriter.create<LLVM::AllocaOp>(
        op.getLoc(), ptrType, rewriter.getI8Type(), bufBytesCst, 1);
    auto allocBuf = allocOp.getResult();

    // Write a zero terminator to the last position
    auto lastPtrOp = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), ptrType, rewriter.getI8Type(), allocBuf,
        LLVM::GEPArg(bufBytes - 1), true);
    auto cst0I8 =
        rewriter.createOrFold<LLVM::ConstantOp>(op.getLoc(), i8Type, 0);
    rewriter.create<LLVM::StoreOp>(op.getLoc(), cst0I8, lastPtrOp);

    // Convert negative values to positives. We take care of the sign later.
    auto iterVal = adaptor.getValue();
    if (op.getIsSigned()) {
      iterVal = rewriter.createOrFold<LLVM::SExtOp>(
          op.getLoc(), rewriter.getIntegerType(width + 1), iterVal);
      iterVal = rewriter.createOrFold<LLVM::AbsOp>(
          op.getLoc(), iterVal.getType(), iterVal, true);
    }

    // Do-while loop for decimal printing. Writes to the allocated buffer back
    // to front. First argument: Value to print Second argument: Byte offset to
    // write the value's least significant decimal digit to
    auto digitsCst = rewriter.createOrFold<LLVM::ConstantOp>(
        op.getLoc(), i32Type, bufBytes - 2);
    auto printWhileOp = rewriter.create<scf::WhileOp>(
        op.getLoc(), TypeRange{iterVal.getType(), digitsCst.getType()},
        ValueRange{iterVal, digitsCst}, [&](auto, auto, auto) {},
        [&](auto, auto, auto) {});

    {
      OpBuilder::InsertionGuard g(rewriter);
      // --- Before block ---
      rewriter.setInsertionPointToStart(printWhileOp.getBeforeBody());
      auto cst10Wide = rewriter.createOrFold<LLVM::ConstantOp>(
          op.getLoc(), iterVal.getType(), 10);
      auto cst0Wide = rewriter.createOrFold<LLVM::ConstantOp>(
          op.getLoc(), iterVal.getType(), 0);

      // Calculate remainder and division by 10.
      auto divOp = rewriter.createOrFold<LLVM::UDivOp>(
          op.getLoc(), printWhileOp.getBeforeArguments()[0], cst10Wide);
      auto remainder = rewriter.createOrFold<LLVM::URemOp>(
          op.getLoc(), printWhileOp.getBeforeArguments()[0], cst10Wide);
      remainder =
          rewriter.createOrFold<LLVM::TruncOp>(op.getLoc(), i8Type, remainder);

      // Calculate the ASCII value based on the remainder and store
      // it at the current offset.
      auto cstChar0 = rewriter.createOrFold<LLVM::ConstantOp>(
          op.getLoc(), i8Type, static_cast<int64_t>('0'));
      auto charOp = rewriter.createOrFold<LLVM::AddOp>(op.getLoc(), i8Type,
                                                       remainder, cstChar0);
      auto ptrOp = rewriter.createOrFold<LLVM::GEPOp>(
          op.getLoc(), ptrType, rewriter.getI8Type(), allocBuf,
          printWhileOp.getBeforeArguments()[1], true);
      rewriter.create<LLVM::StoreOp>(op.getLoc(), charOp, ptrOp);

      // Terminate when the division has reached zero.
      auto doIter = rewriter.createOrFold<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::ne, divOp, cst0Wide);
      rewriter.create<scf::ConditionOp>(
          op.getLoc(), doIter,
          ValueRange{divOp, printWhileOp.getBeforeArguments()[1]});

      // --- After block ---
      rewriter.setInsertionPointToStart(printWhileOp.getAfterBody());
      // Reduce the write offset by one and iterate with the divided value.
      auto cst1I32 =
          rewriter.createOrFold<LLVM::ConstantOp>(op.getLoc(), i32Type, 1);
      auto idxSubOp = rewriter.createOrFold<LLVM::SubOp>(
          op.getLoc(), i32Type, printWhileOp.getAfterArguments()[1], cst1I32);
      rewriter.create<scf::YieldOp>(
          op.getLoc(),
          ValueRange{printWhileOp.getAfterArguments()[0], idxSubOp});
    }

    // Returns the last offset that has been written to
    auto printOffset = printWhileOp.getResult(1);

    // If the original value was negative, prepend a minus sign
    // and return the updated offset.
    if (op.getIsSigned()) {
      auto cst0Wide = rewriter.createOrFold<LLVM::ConstantOp>(
          op.getLoc(), adaptor.getValue().getType(), 0);
      auto isNegative = rewriter.createOrFold<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::slt, adaptor.getValue(), cst0Wide);
      auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), i32Type, isNegative,
                                             true, true);

      // -- Then block --
      auto thenBuilder = ifOp.getThenBodyBuilder();
      auto cst1I32 =
          thenBuilder.createOrFold<LLVM::ConstantOp>(op.getLoc(), i32Type, 1);
      auto subOffset = thenBuilder.createOrFold<LLVM::SubOp>(
          op.getLoc(), i32Type, printOffset, cst1I32);
      auto ptrOp = thenBuilder.createOrFold<LLVM::GEPOp>(
          op.getLoc(), ptrType, rewriter.getI8Type(), allocBuf, subOffset,
          true);
      auto cstCharMinus = thenBuilder.createOrFold<LLVM::ConstantOp>(
          op.getLoc(), i8Type, static_cast<int64_t>('-'));
      thenBuilder.create<LLVM::StoreOp>(op.getLoc(), cstCharMinus, ptrOp);
      thenBuilder.create<scf::YieldOp>(op.getLoc(), subOffset);

      // -- Else block --
      ifOp.getElseBodyBuilder().create<scf::YieldOp>(op.getLoc(), printOffset);

      printOffset = ifOp.getResult(0);
    }

    // While loop to fill the remaining positions with spaces back to front.
    // Argument: Byte offset that has been written to previously.

    auto padWhileOp = rewriter.create<scf::WhileOp>(
        op.getLoc(), TypeRange{i32Type}, ValueRange{printOffset},
        [&](auto, auto, auto) {}, [&](auto, auto, auto) {});

    {
      OpBuilder::InsertionGuard g(rewriter);
      // --- Before block ---
      rewriter.setInsertionPointToStart(padWhileOp.getBeforeBody());

      // Terminate if we have reached the start of the buffer.
      auto cst0I32 =
          rewriter.createOrFold<LLVM::ConstantOp>(op.getLoc(), i32Type, 0);
      auto isZeroOp = rewriter.createOrFold<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::ne,
          padWhileOp.getBeforeArguments()[0], cst0I32);
      rewriter.create<scf::ConditionOp>(
          op.getLoc(), isZeroOp,
          ValueRange{padWhileOp.getBeforeArguments()[0]});

      // --- After block ---
      rewriter.setInsertionPointToStart(padWhileOp.getAfterBody());

      // Reduce the offset by one and write a space to its position.
      auto cst1I32 =
          rewriter.createOrFold<LLVM::ConstantOp>(op.getLoc(), i32Type, 1);
      auto subOffset = rewriter.createOrFold<LLVM::SubOp>(
          op.getLoc(), i32Type, padWhileOp.getAfterArguments()[0], cst1I32);
      auto cstCharSpace = rewriter.createOrFold<LLVM::ConstantOp>(
          op.getLoc(), i8Type, static_cast<int64_t>(' '));
      auto ptrOp = rewriter.createOrFold<LLVM::GEPOp>(
          op.getLoc(), ptrType, rewriter.getI8Type(), allocBuf, subOffset,
          true);
      rewriter.create<LLVM::StoreOp>(op.getLoc(), cstCharSpace, ptrOp);

      rewriter.create<scf::YieldOp>(op.getLoc(), subOffset);
    }

    rewriter.replaceOp(op, allocOp);
    return success();
  }
};

struct FormatBinOpLowering : public OpConversionPattern<sim::FormatBinOp> {

  using OpConversionPattern::OpConversionPattern;

  // Create the binary string representation of an arbitray integer
  LogicalResult
  matchAndRewrite(sim::FormatBinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Type = rewriter.getI32Type();
    auto i8Type = rewriter.getI8Type();

    // Allocate the reqired number of characters (including zero terminator) on
    // the stack.
    unsigned width = adaptor.getValue().getType().getIntOrFloatBitWidth();
    auto widthPlusOneCst = rewriter.createOrFold<LLVM::ConstantOp>(
        op.getLoc(), i32Type, width + 1);
    auto allocBuf = rewriter.createOrFold<LLVM::AllocaOp>(
        op.getLoc(), ptrType, rewriter.getI8Type(), widthPlusOneCst, 1);

    // Write a zero terminator to the last position
    auto lastPtrOp =
        rewriter.create<LLVM::GEPOp>(op.getLoc(), ptrType, rewriter.getI8Type(),
                                     allocBuf, LLVM::GEPArg(width), true);
    auto cst0I8 =
        rewriter.createOrFold<LLVM::ConstantOp>(op.getLoc(), i8Type, 0);
    rewriter.create<LLVM::StoreOp>(op.getLoc(), cst0I8, lastPtrOp);

    auto cst0I32 =
        rewriter.createOrFold<LLVM::ConstantOp>(op.getLoc(), i32Type, 0);
    auto cst1I32 =
        rewriter.createOrFold<LLVM::ConstantOp>(op.getLoc(), i32Type, 1);
    auto widthCst =
        rewriter.createOrFold<LLVM::ConstantOp>(op.getLoc(), i32Type, width);
    auto widthMinusOneCst = rewriter.createOrFold<LLVM::ConstantOp>(
        op.getLoc(), i32Type, width - 1);

    // For-loop filling the buffer with 0/1 characters front to back.
    // Iterates on the number of characters written.
    auto forOp =
        rewriter.create<scf::ForOp>(op.getLoc(), cst0I32, widthCst, cst1I32);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      auto cst0Wide = rewriter.createOrFold<LLVM::ConstantOp>(
          op.getLoc(), adaptor.getValue().getType(), 0);
      auto cst1Wide = rewriter.createOrFold<LLVM::ConstantOp>(
          op.getLoc(), adaptor.getValue().getType(), 1);
      auto cst0Char =
          rewriter.createOrFold<LLVM::ConstantOp>(op.getLoc(), i8Type, '0');
      auto cst1Char =
          rewriter.createOrFold<LLVM::ConstantOp>(op.getLoc(), i8Type, '1');
      auto flags =
          LLVM::IntegerOverflowFlags::nuw | LLVM::IntegerOverflowFlags::nsw;

      // Calcualate the current digit position: width - 1 - i
      auto bitPos = rewriter.createOrFold<LLVM::SubOp>(
          op.getLoc(), widthMinusOneCst, forOp.getInductionVar(), flags);

      // Create a bit mask isolating the current digit.
      if (width > 32) {
        bitPos = rewriter.createOrFold<LLVM::ZExtOp>(
            op.getLoc(), adaptor.getValue().getType(), bitPos);
      } else if (width < 32) {
        bitPos = rewriter.createOrFold<LLVM::TruncOp>(
            op.getLoc(), adaptor.getValue().getType(), bitPos);
      }
      auto mask = rewriter.createOrFold<LLVM::ShlOp>(
          op.getLoc(), adaptor.getValue().getType(), cst1Wide, bitPos, flags);

      // Select the corresponding character for the current posiiton.
      auto andOp = rewriter.createOrFold<LLVM::AndOp>(op.getLoc(),
                                                      adaptor.getValue(), mask);
      auto eqOp = rewriter.createOrFold<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::ne, andOp, cst0Wide);
      auto charSelOp = rewriter.create<LLVM::SelectOp>(op.getLoc(), eqOp,
                                                       cst1Char, cst0Char);

      // Store at the offset.
      auto ptrOp = rewriter.create<LLVM::GEPOp>(op.getLoc(), ptrType,
                                                rewriter.getI8Type(), allocBuf,
                                                forOp.getInductionVar(), true);
      rewriter.create<LLVM::StoreOp>(op.getLoc(), charSelOp, ptrOp);
    }

    rewriter.replaceOp(op, allocBuf);
    return success();
  }
};

} // namespace

LLVM::GlobalOp
LowerPrintsToArcEnvCallsPass::lookupOrCreateStringSymbol(RewriterBase &rewriter,
                                                         StringAttr str) {
  auto &cached = stringSymbolCache[str];
  if (cached)
    return cached;

  auto symName = "_arc_fmt_string_" + Twine(stringSymbolCache.size());
  IRRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(getOperation().getBody());
  auto globalType = LLVM::LLVMArrayType::get(rewriter.getI8Type(), str.size());
  cached = rewriter.create<LLVM::GlobalOp>(
      getOperation().getLoc(), globalType, /*isConstant=*/true,
      LLVM::Linkage::Internal,
      /*name=*/rewriter.getStringAttr(symName), str,
      /*alignment=*/0);
  return cached;
}

// Check if the given operation is a format string token that can be inlined
// in a C-style format string.
static bool isInlinableFormat(Operation *op) {
  if (!op)
    return false;
  if (auto binFmt = dyn_cast<sim::FormatBinOp>(op)) {
    auto width = binFmt.getValue().getType().getIntOrFloatBitWidth();
    return width == 1;
  }
  if (auto decFmt = dyn_cast<sim::FormatDecOp>(op)) {
    auto width = decFmt.getValue().getType().getIntOrFloatBitWidth();
    return width <= 64;
  }
  return isa<sim::FormatLitOp, sim::FormatHexOp, sim::FormatCharOp>(op);
}

// Formatting Helper of Doom:
// Convert a format fragment to a libc compatible formatting string.
static void inlineFormatFragment(OpBuilder &builder, Operation *fragmentOp,
                                 bool singleOperand, SmallString<32> &fmtString,
                                 SmallVectorImpl<Value> &arguments,
                                 bool &usePutc, bool &usePuts) {

  auto loc = fragmentOp->getLoc();
  TypeSwitch<Operation *>(fragmentOp)
      .Case<sim::FormatLitOp>([&](sim::FormatLitOp fmtOp) {
        if (singleOperand) {
          usePutc = fmtOp.getLiteral().size() == 1;
          usePuts = !usePutc;
        }
        // Escape the string as required
        for (auto c : fmtOp.getLiteral()) {
          if (c == '\0')
            fragmentOp->emitWarning("String containig NULL character will not "
                                    "be printed correctly.");
          if (c == '%' && !usePuts && !usePutc)
            fmtString += "%%";
          else
            fmtString.push_back(c);
        }
      })
      .Case<sim::FormatHexOp>([&](sim::FormatHexOp fmtOp) {
        // Split hexadecimal substitutions into chunks of up to 64 bits
        auto width = fmtOp.getValue().getType().getIntOrFloatBitWidth();
        auto remainingWidth = width;
        unsigned chunkWidth =
            remainingWidth % 64 == 0 ? 64 : remainingWidth % 64;
        while (remainingWidth != 0) {
          unsigned numDigits = chunkWidth / 4;
          if (chunkWidth % 4 != 0)
            numDigits++;
          fmtString += "%0";
          fmtString += std::to_string(numDigits);
          fmtString += "llx";
          Value argValue = fmtOp.getValue();
          if (remainingWidth - chunkWidth != 0) {
            auto shiftCst = builder.createOrFold<LLVM::ConstantOp>(
                loc, fmtOp.getValue().getType(), remainingWidth - chunkWidth);
            argValue = builder.createOrFold<LLVM::LShrOp>(loc, fmtOp.getValue(),
                                                          shiftCst);
          }
          if (width < 64)
            argValue = builder.createOrFold<LLVM::ZExtOp>(
                loc, builder.getI64Type(), argValue);
          else if (width > 64)
            argValue = builder.createOrFold<LLVM::TruncOp>(
                loc, builder.getI64Type(), argValue);
          arguments.push_back(argValue);
          remainingWidth -= chunkWidth;
          chunkWidth = 64;
        }
      })
      .Case<sim::FormatDecOp>([&](sim::FormatDecOp fmtOp) {
        Value argVal = fmtOp.getValue();
        auto width = fmtOp.getValue().getType().getIntOrFloatBitWidth();
        auto widthOp = builder.createOrFold<LLVM::ConstantOp>(
            loc, builder.getI32Type(),
            sim::FormatDecOp::getDecimalWidth(width, fmtOp.getIsSigned()));
        arguments.push_back(widthOp);
        if (fmtOp.getIsSigned()) {
          fmtString += "%*lld";
          if (width < 64)
            argVal = builder.createOrFold<LLVM::SExtOp>(
                loc, builder.getI64Type(), fmtOp.getValue());
        } else {
          fmtString += "%*llu";
          if (width < 64)
            argVal = builder.createOrFold<LLVM::ZExtOp>(
                loc, builder.getI64Type(), fmtOp.getValue());
        }
        arguments.push_back(argVal);
      })
      .Case<sim::FormatBinOp>([&](sim::FormatBinOp fmtOp) {
        assert(fmtOp.getValue().getType() == builder.getI1Type());
        // Inline single bit binary substitutions as character.
        if (singleOperand)
          usePutc = true;
        auto cst0 = builder.createOrFold<LLVM::ConstantOp>(
            loc, builder.getI32Type(), '0');
        auto cst1 = builder.createOrFold<LLVM::ConstantOp>(
            loc, builder.getI32Type(), '1');
        auto selOp = builder.createOrFold<LLVM::SelectOp>(loc, fmtOp.getValue(),
                                                          cst1, cst0);
        fmtString += "%c";
        arguments.push_back(selOp);
      })
      .Case<sim::FormatCharOp>([&](sim::FormatCharOp fmtOp) {
        if (singleOperand)
          usePutc = true;
        fmtString += "%c";
        Value valOp = fmtOp.getValue();
        auto width = fmtOp.getValue().getType().getIntOrFloatBitWidth();
        if (width > 8) {
          fmtOp.emitWarning("Truncating char format substitution to 8 bit");
          valOp = builder.createOrFold<LLVM::TruncOp>(loc, builder.getI8Type(),
                                                      fmtOp.getValue());
        }
        auto extOp = builder.createOrFold<LLVM::ZExtOp>(
            loc, builder.getI32Type(), valOp);
        arguments.push_back(extOp);
      })
      .Default([&](auto fmtOp) {
        assert(false && "Unable to inline format string.");
      });
}

LogicalResult LowerPrintsToArcEnvCallsPass::convertPrintOp(
    RewriterBase &rewriter, sim::PrintFormattedProcOp printOp) {
  SmallString<32> fmtString;
  SmallVector<Value> fragments;
  rewriter.setInsertionPoint(printOp);

  if (auto concatOp =
          printOp.getInput().getDefiningOp<sim::FormatStringConcatOp>()) {
    auto isAcyclic = concatOp.getFlattenedInputs(fragments);
    if (failed(isAcyclic)) {
      printOp.emitError("Cyclic format string detected.");
      return failure();
    }
  } else {
    fragments.emplace_back(printOp.getInput());
  }

  if (fragments.empty()) {
    rewriter.eraseOp(printOp);
    return success();
  }

  bool usePuts = false;
  bool usePutc = false;
  bool singleOperand = (fragments.size() == 1);

  SmallVector<Value> arguments;
  arguments.reserve(fragments.size());

  for (auto fragment : fragments) {
    auto *defOp = fragment.getDefiningOp();
    if (!defOp) {
      printOp.emitError(
          "Cannot lower format string fragment passed as argument.");
      return failure();
    }
    if (!isInlinableFormat(defOp)) {
      // Unsupported formattings are stringified seperately and passed as
      // char*
      fmtString += "%s";
      auto fmtToPtrCast = rewriter.create<UnrealizedConversionCastOp>(
          defOp->getLoc(), LLVM::LLVMPointerType::get(rewriter.getContext()),
          fragment);
      arguments.push_back(fmtToPtrCast.getResult(0));
      if (singleOperand)
        usePuts = true;
      continue;
    }

    inlineFormatFragment(rewriter, defOp, singleOperand, fmtString, arguments,
                         usePutc, usePuts);
  }
  assert(!(usePutc && usePuts));
  assert(!(usePutc || usePuts) || arguments.size() <= 1);
  envCallInfo.putcUsed |= usePutc;
  envCallInfo.putsUsed |= usePuts;
  envCallInfo.printfUsed |= !(usePutc || usePuts);

  auto int32Type = rewriter.getI32Type();

  auto cst0 =
      rewriter.createOrFold<LLVM::ConstantOp>(printOp.getLoc(), int32Type, 0);
  auto getPrintStreamCall = rewriter.create<LLVM::CallOp>(
      printOp.getLoc(), envCallInfo.getPrintStreamFunc, cst0);

  SmallVector<Value> callArgs;

  if (usePutc) {
    if (arguments.empty())
      callArgs.emplace_back(rewriter.createOrFold<LLVM::ConstantOp>(
          printOp.getLoc(), int32Type, fmtString[0]));
    else
      callArgs.emplace_back(arguments.front());
    callArgs.emplace_back(getPrintStreamCall.getResult());
    rewriter.create<LLVM::CallOp>(printOp.getLoc(), envCallInfo.putcFunc,
                                  callArgs);
    rewriter.eraseOp(printOp);
    return success();
  }

  if (usePuts && !arguments.empty()) {
    assert(arguments.size() == 1);
    // Dynamic 'fputs'
    callArgs.emplace_back(arguments.front());
    callArgs.emplace_back(getPrintStreamCall.getResult());
    rewriter.create<LLVM::CallOp>(printOp.getLoc(), envCallInfo.putsFunc,
                                  callArgs);
    rewriter.eraseOp(printOp);
    return success();
  }

  fmtString.emplace_back('\0');
  auto fmtStringGlobalOp =
      lookupOrCreateStringSymbol(rewriter, rewriter.getStringAttr(fmtString));
  auto fmtStringAddrOp = rewriter.createOrFold<LLVM::AddressOfOp>(
      printOp.getLoc(), fmtStringGlobalOp);

  if (usePuts) {
    assert(arguments.empty());
    // Static 'fputs'
    callArgs.emplace_back(fmtStringAddrOp);
    callArgs.emplace_back(getPrintStreamCall.getResult());
    rewriter.create<LLVM::CallOp>(printOp.getLoc(), envCallInfo.putsFunc,
                                  callArgs);
    rewriter.eraseOp(printOp);
    return success();
  }

  callArgs.reserve(arguments.size() + 2);
  callArgs.emplace_back(getPrintStreamCall.getResult());
  callArgs.emplace_back(fmtStringAddrOp);
  callArgs.append(arguments.begin(), arguments.end());
  rewriter.create<LLVM::CallOp>(printOp.getLoc(), envCallInfo.printfFunc,
                                callArgs);
  rewriter.eraseOp(printOp);
  return success();
}

LogicalResult LowerPrintsToArcEnvCallsPass::lowerPrints(
    RewriterBase &rewriter, ArrayRef<sim::PrintFormattedProcOp> printOps) {
  SmallVector<Value> cleanupList;

  bool hasFailed = false;
  for (auto printOp : printOps) {
    cleanupList.emplace_back(printOp.getInput());
    if (failed(convertPrintOp(rewriter, printOp)))
      hasFailed = true;
  }

  if (hasFailed)
    return failure();

  SmallVector<Value> worklist;
  while (!cleanupList.empty()) {
    bool hasChanged = false;
    worklist = std::move(cleanupList);
    cleanupList = SmallVector<Value>();

    for (auto workItem : worklist) {
      if (!workItem || !workItem.getDefiningOp())
        continue;
      if (!workItem.use_empty()) {
        cleanupList.emplace_back(workItem);
        continue;
      }
      if (auto concat = workItem.getDefiningOp<sim::FormatStringConcatOp>())
        cleanupList.append(concat.getInputs().begin(),
                           concat.getInputs().end());
      rewriter.eraseOp(workItem.getDefiningOp());
      hasChanged = true;
    }

    if (!hasChanged)
      break;
  }

  if (cleanupList.empty())
    return success();

  LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<scf::SCFDialect>();

  target.addIllegalOp<sim::FormatBinOp, sim::FormatDecOp>();
  LLVMTypeConverter converter(&getContext());
  converter.addConversion([&](sim::FormatStringType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<FormatDecOpLowering, FormatBinOpLowering>(converter,
                                                         &getContext());

  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

void LowerPrintsToArcEnvCallsPass::runOnOperation() {
  envCallInfo = ArcEnvCallInfo();

  SmallVector<sim::PrintFormattedProcOp> printOps;

  auto theModule = getOperation();

  theModule.walk([&](sim::PrintFormattedProcOp printOp) {
    printOps.emplace_back(printOp);
  });

  if (printOps.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  IRRewriter rewriter(theModule);

  auto ptrType = LLVM::LLVMPointerType::get(theModule.getContext());
  auto int32Type = rewriter.getI32Type();
  envCallInfo.getPrintStreamFunc = lookupOrInsertExternalFunction(
      rewriter, theModule, "_arc_env_get_print_stream",
      LLVM::LLVMFunctionType::get(ptrType, {int32Type}));
  envCallInfo.printfFunc = lookupOrInsertExternalFunction(
      rewriter, theModule, "_arc_libc_fprintf",
      LLVM::LLVMFunctionType::get(int32Type, {ptrType, ptrType}, true));
  envCallInfo.putsFunc = lookupOrInsertExternalFunction(
      rewriter, theModule, "_arc_libc_fputs",
      LLVM::LLVMFunctionType::get(int32Type, {ptrType, ptrType}));
  envCallInfo.putcFunc = lookupOrInsertExternalFunction(
      rewriter, theModule, "_arc_libc_fputc",
      LLVM::LLVMFunctionType::get(int32Type, {int32Type, ptrType}));

  if (failed(lowerPrints(rewriter, printOps))) {
    signalPassFailure();
    return;
  }

  if (!envCallInfo.printfUsed)
    rewriter.eraseOp(envCallInfo.printfFunc);
  if (!envCallInfo.putsUsed)
    rewriter.eraseOp(envCallInfo.putsFunc);
  if (!envCallInfo.putcUsed)
    rewriter.eraseOp(envCallInfo.putcFunc);
}
