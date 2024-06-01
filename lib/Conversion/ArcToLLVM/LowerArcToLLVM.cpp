//===- LowerArcToLLVM.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/ArcToLLVM.h"
#include "circt/Conversion/CombToLLVM.h"
#include "circt/Conversion/HWToLLVM.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ModelInfo.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-arc-to-llvm"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

static llvm::Twine evalSymbolFromModelName(StringRef modelName) {
  return modelName + "_eval";
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

static constexpr StringLiteral arcEnvSymGetPrintStream =
    "_arc_env_get_print_stream";

static constexpr StringLiteral arcStlSymFprintf = "_arc_stl_fprintf";

static constexpr StringLiteral arcStlSymFputs = "_arc_stl_fputs";

static constexpr StringLiteral arcStlSymFputc = "_arc_stl_fputc";

// Lookup or insert a global declaration of an external function to the module
// toplevel
static LLVM::LLVMFuncOp
lookupOrInsertExternalFunction(OpBuilder &builder, mlir::ModuleOp module,
                               StringRef symbolName,
                               LLVM::LLVMFunctionType fnType) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(symbolName);
  if (func)
    return func;

  auto unknownLoc = UnknownLoc::get(module.getContext());
  func = builder.create<LLVM::LLVMFuncOp>(unknownLoc, symbolName, fnType,
                                          LLVM::Linkage::External,
                                          /*dsoLocal=*/false, LLVM::CConv::C);
  return func;
}

namespace {

struct ModelOpLowering : public OpConversionPattern<arc::ModelOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::ModelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    {
      IRRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&op.getBodyBlock());
      rewriter.create<func::ReturnOp>(op.getLoc());
    }
    auto funcName =
        rewriter.getStringAttr(evalSymbolFromModelName(op.getName()));
    auto funcType =
        rewriter.getFunctionType(op.getBody().getArgumentTypes(), {});
    auto func =
        rewriter.create<mlir::func::FuncOp>(op.getLoc(), funcName, funcType);
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct AllocStorageOpLowering
    : public OpConversionPattern<arc::AllocStorageOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::AllocStorageOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto type = typeConverter->convertType(op.getType());
    if (!op.getOffset().has_value())
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, type, rewriter.getI8Type(),
                                             adaptor.getInput(),
                                             LLVM::GEPArg(*op.getOffset()));
    return success();
  }
};

template <class ConcreteOp>
struct AllocStateLikeOpLowering : public OpConversionPattern<ConcreteOp> {
  using OpConversionPattern<ConcreteOp>::OpConversionPattern;
  using OpConversionPattern<ConcreteOp>::typeConverter;
  using OpAdaptor = typename ConcreteOp::Adaptor;

  LogicalResult
  matchAndRewrite(ConcreteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Get a pointer to the correct offset in the storage.
    auto offsetAttr = op->template getAttrOfType<IntegerAttr>("offset");
    if (!offsetAttr)
      return failure();
    Value ptr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), adaptor.getStorage().getType(), rewriter.getI8Type(),
        adaptor.getStorage(),
        LLVM::GEPArg(offsetAttr.getValue().getZExtValue()));
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct StateReadOpLowering : public OpConversionPattern<arc::StateReadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::StateReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto type = typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, type, adaptor.getState());
    return success();
  }
};

struct StateWriteOpLowering : public OpConversionPattern<arc::StateWriteOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::StateWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getCondition()) {
      rewriter.replaceOpWithNewOp<scf::IfOp>(
          op, adaptor.getCondition(), [&](auto &builder, auto loc) {
            builder.template create<LLVM::StoreOp>(loc, adaptor.getValue(),
                                                   adaptor.getState());
            builder.template create<scf::YieldOp>(loc);
          });
    } else {
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                                 adaptor.getState());
    }
    return success();
  }
};

struct AllocMemoryOpLowering : public OpConversionPattern<arc::AllocMemoryOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::AllocMemoryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto offsetAttr = op->getAttrOfType<IntegerAttr>("offset");
    if (!offsetAttr)
      return failure();
    Value ptr = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), adaptor.getStorage().getType(), rewriter.getI8Type(),
        adaptor.getStorage(),
        LLVM::GEPArg(offsetAttr.getValue().getZExtValue()));

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct StorageGetOpLowering : public OpConversionPattern<arc::StorageGetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::StorageGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value offset = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), op.getOffsetAttr());
    Value ptr = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), adaptor.getStorage().getType(), rewriter.getI8Type(),
        adaptor.getStorage(), offset);
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct MemoryAccess {
  Value ptr;
  Value withinBounds;
};

static MemoryAccess prepareMemoryAccess(Location loc, Value memory,
                                        Value address, MemoryType type,
                                        ConversionPatternRewriter &rewriter) {
  auto zextAddrType = rewriter.getIntegerType(
      cast<IntegerType>(address.getType()).getWidth() + 1);
  Value addr = rewriter.create<LLVM::ZExtOp>(loc, zextAddrType, address);
  Value addrLimit = rewriter.create<LLVM::ConstantOp>(
      loc, zextAddrType, rewriter.getI32IntegerAttr(type.getNumWords()));
  Value withinBounds = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::ult, addr, addrLimit);
  Value ptr = rewriter.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(memory.getContext()),
      rewriter.getIntegerType(type.getStride() * 8), memory, ValueRange{addr});
  return {ptr, withinBounds};
}

struct MemoryReadOpLowering : public OpConversionPattern<arc::MemoryReadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::MemoryReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto type = typeConverter->convertType(op.getType());
    auto memoryType = cast<MemoryType>(op.getMemory().getType());
    auto access =
        prepareMemoryAccess(op.getLoc(), adaptor.getMemory(),
                            adaptor.getAddress(), memoryType, rewriter);

    // Only attempt to read the memory if the address is within bounds,
    // otherwise produce a zero value.
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, access.withinBounds,
        [&](auto &builder, auto loc) {
          Value loadOp = builder.template create<LLVM::LoadOp>(
              loc, memoryType.getWordType(), access.ptr);
          builder.template create<scf::YieldOp>(loc, loadOp);
        },
        [&](auto &builder, auto loc) {
          Value zeroValue = builder.template create<LLVM::ConstantOp>(
              loc, type, builder.getI64IntegerAttr(0));
          builder.template create<scf::YieldOp>(loc, zeroValue);
        });
    return success();
  }
};

struct MemoryWriteOpLowering : public OpConversionPattern<arc::MemoryWriteOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::MemoryWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto access = prepareMemoryAccess(
        op.getLoc(), adaptor.getMemory(), adaptor.getAddress(),
        cast<MemoryType>(op.getMemory().getType()), rewriter);
    auto enable = access.withinBounds;
    if (adaptor.getEnable())
      enable = rewriter.create<LLVM::AndOp>(op.getLoc(), adaptor.getEnable(),
                                            enable);

    // Only attempt to write the memory if the address is within bounds.
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, enable, [&](auto &builder, auto loc) {
          builder.template create<LLVM::StoreOp>(loc, adaptor.getData(),
                                                 access.ptr);
          builder.template create<scf::YieldOp>(loc);
        });
    return success();
  }
};

/// A dummy lowering for clock gates to an AND gate.
struct ClockGateOpLowering : public OpConversionPattern<seq::ClockGateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(seq::ClockGateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, adaptor.getInput(),
                                             adaptor.getEnable(), true);
    return success();
  }
};

struct ZeroCountOpLowering : public OpConversionPattern<arc::ZeroCountOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::ZeroCountOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Use poison when input is zero.
    IntegerAttr isZeroPoison = rewriter.getBoolAttr(true);

    if (op.getPredicate() == arc::ZeroCountPredicate::leading) {
      rewriter.replaceOpWithNewOp<LLVM::CountLeadingZerosOp>(
          op, adaptor.getInput().getType(), adaptor.getInput(), isZeroPoison);
      return success();
    }

    rewriter.replaceOpWithNewOp<LLVM::CountTrailingZerosOp>(
        op, adaptor.getInput().getType(), adaptor.getInput(), isZeroPoison);
    return success();
  }
};

struct SeqConstClockLowering : public OpConversionPattern<seq::ConstClockOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(seq::ConstClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, rewriter.getI1Type(), static_cast<int64_t>(op.getValue()));
    return success();
  }
};

template <typename OpTy>
struct ReplaceOpWithInputPattern : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Simulation Orchestration Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

struct ModelInfoMap {
  size_t numStateBytes;
  llvm::DenseMap<StringRef, StateInfo> states;
};

template <typename OpTy>
struct ModelAwarePattern : public OpConversionPattern<OpTy> {
  ModelAwarePattern(const TypeConverter &typeConverter, MLIRContext *context,
                    llvm::DenseMap<StringRef, ModelInfoMap> &modelInfo)
      : OpConversionPattern<OpTy>(typeConverter, context),
        modelInfo(modelInfo) {}

protected:
  Value createPtrToPortState(ConversionPatternRewriter &rewriter, Location loc,
                             Value state, const StateInfo &port) const {
    MLIRContext *ctx = rewriter.getContext();
    return rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(ctx),
                                        IntegerType::get(ctx, 8), state,
                                        LLVM::GEPArg(port.offset));
  }

  llvm::DenseMap<StringRef, ModelInfoMap> &modelInfo;
};

/// Lowers SimInstantiateOp to a malloc and memset call. This pattern will
/// mutate the global module.
struct SimInstantiateOpLowering
    : public ModelAwarePattern<arc::SimInstantiateOp> {
  using ModelAwarePattern::ModelAwarePattern;

  LogicalResult
  matchAndRewrite(arc::SimInstantiateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto modelIt = modelInfo.find(
        cast<SimModelInstanceType>(op.getBody().getArgument(0).getType())
            .getModel()
            .getValue());
    ModelInfoMap &model = modelIt->second;

    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    ConversionPatternRewriter::InsertionGuard guard(rewriter);

    // FIXME: like the rest of MLIR, this assumes sizeof(intptr_t) ==
    // sizeof(size_t) on the target architecture.
    Type convertedIndex = typeConverter->convertType(rewriter.getIndexType());

    LLVM::LLVMFuncOp mallocFunc =
        LLVM::lookupOrCreateMallocFn(moduleOp, convertedIndex);
    LLVM::LLVMFuncOp freeFunc = LLVM::lookupOrCreateFreeFn(moduleOp);

    Location loc = op.getLoc();
    Value numStateBytes = rewriter.create<LLVM::ConstantOp>(
        loc, convertedIndex, model.numStateBytes);
    Value allocated =
        rewriter
            .create<LLVM::CallOp>(loc, mallocFunc, ValueRange{numStateBytes})
            .getResult();
    Value zero =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8Type(), 0);
    rewriter.create<LLVM::MemsetOp>(loc, allocated, zero, numStateBytes, false);
    rewriter.inlineBlockBefore(&adaptor.getBody().getBlocks().front(), op,
                               {allocated});
    rewriter.create<LLVM::CallOp>(loc, freeFunc, ValueRange{allocated});
    rewriter.eraseOp(op);

    return success();
  }
};

struct SimSetInputOpLowering : public ModelAwarePattern<arc::SimSetInputOp> {
  using ModelAwarePattern::ModelAwarePattern;

  LogicalResult
  matchAndRewrite(arc::SimSetInputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto modelIt =
        modelInfo.find(cast<SimModelInstanceType>(op.getInstance().getType())
                           .getModel()
                           .getValue());
    ModelInfoMap &model = modelIt->second;

    auto portIt = model.states.find(op.getInput());
    if (portIt == model.states.end()) {
      // If the port is not found in the state, it means the model does not
      // actually use it. Thus this operation is a no-op.
      rewriter.eraseOp(op);
      return success();
    }

    StateInfo &port = portIt->second;
    Value statePtr = createPtrToPortState(rewriter, op.getLoc(),
                                          adaptor.getInstance(), port);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                               statePtr);

    return success();
  }
};

struct SimGetPortOpLowering : public ModelAwarePattern<arc::SimGetPortOp> {
  using ModelAwarePattern::ModelAwarePattern;

  LogicalResult
  matchAndRewrite(arc::SimGetPortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto modelIt =
        modelInfo.find(cast<SimModelInstanceType>(op.getInstance().getType())
                           .getModel()
                           .getValue());
    ModelInfoMap &model = modelIt->second;

    auto portIt = model.states.find(op.getPort());
    if (portIt == model.states.end()) {
      // If the port is not found in the state, it means the model does not
      // actually set it. Thus this operation returns 0.
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, typeConverter->convertType(op.getValue().getType()), 0);
      return success();
    }

    StateInfo &port = portIt->second;
    Value statePtr = createPtrToPortState(rewriter, op.getLoc(),
                                          adaptor.getInstance(), port);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getValue().getType(),
                                              statePtr);

    return success();
  }
};

struct SimStepOpLowering : public ModelAwarePattern<arc::SimStepOp> {
  using ModelAwarePattern::ModelAwarePattern;

  LogicalResult
  matchAndRewrite(arc::SimStepOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    StringRef modelName = cast<SimModelInstanceType>(op.getInstance().getType())
                              .getModel()
                              .getValue();

    StringAttr evalFunc =
        rewriter.getStringAttr(evalSymbolFromModelName(modelName));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, std::nullopt, evalFunc,
                                              adaptor.getInstance());

    return success();
  }
};

/// Lowers SimEmitValueOp to a printf call. The integer will be printed in its
/// entirety if it is of size up to size_t, and explicitly truncated otherwise.
/// This pattern will mutate the global module.
struct SimEmitValueOpLowering
    : public OpConversionPattern<arc::SimEmitValueOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arc::SimEmitValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto valueType = dyn_cast<IntegerType>(adaptor.getValue().getType());
    if (!valueType)
      return failure();

    Location loc = op.getLoc();

    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    // Cast the value to a size_t.
    // FIXME: like the rest of MLIR, this assumes sizeof(intptr_t) ==
    // sizeof(size_t) on the target architecture.
    Value toPrint = adaptor.getValue();
    DataLayout layout = DataLayout::closest(op);
    llvm::TypeSize sizeOfSizeT =
        layout.getTypeSizeInBits(rewriter.getIndexType());
    assert(!sizeOfSizeT.isScalable() &&
           sizeOfSizeT.getFixedValue() <= std::numeric_limits<unsigned>::max());
    bool truncated = false;
    if (valueType.getWidth() > sizeOfSizeT) {
      toPrint = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(getContext(), sizeOfSizeT.getFixedValue()),
          toPrint);
      truncated = true;
    } else if (valueType.getWidth() < sizeOfSizeT)
      toPrint = rewriter.create<LLVM::ZExtOp>(
          loc, IntegerType::get(getContext(), sizeOfSizeT.getFixedValue()),
          toPrint);

    // Lookup of create printf function symbol.
    auto printfFunc = LLVM::lookupOrCreateFn(
        moduleOp, "printf", LLVM::LLVMPointerType::get(getContext()),
        LLVM::LLVMVoidType::get(getContext()), true);

    // Insert the format string if not already available.
    SmallString<16> formatStrName{"_arc_sim_emit_"};
    formatStrName.append(truncated ? "trunc_" : "full_");
    formatStrName.append(adaptor.getValueName());
    LLVM::GlobalOp formatStrGlobal;
    if (!(formatStrGlobal =
              moduleOp.lookupSymbol<LLVM::GlobalOp>(formatStrName))) {
      ConversionPatternRewriter::InsertionGuard insertGuard(rewriter);

      SmallString<16> formatStr = adaptor.getValueName();
      formatStr.append(" = ");
      if (truncated)
        formatStr.append("(truncated) ");
      formatStr.append("%zx\n");
      SmallVector<char> formatStrVec{formatStr.begin(), formatStr.end()};
      formatStrVec.push_back(0);

      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto globalType =
          LLVM::LLVMArrayType::get(rewriter.getI8Type(), formatStrVec.size());
      formatStrGlobal = rewriter.create<LLVM::GlobalOp>(
          loc, globalType, /*isConstant=*/true, LLVM::Linkage::Internal,
          /*name=*/formatStrName, rewriter.getStringAttr(formatStrVec),
          /*alignment=*/0);
    }

    Value formatStrGlobalPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, formatStrGlobal);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, printfFunc, ValueRange{formatStrGlobalPtr, toPrint});

    return success();
  }
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

struct PrintFormattedOpLowering
    : public OpConversionPattern<sim::PrintFormattedProcOp> {

  PrintFormattedOpLowering(const TypeConverter &typeConverter,
                           MLIRContext *context, Namespace &globals)
      : OpConversionPattern<sim::PrintFormattedProcOp>::OpConversionPattern(
            typeConverter, context),
        globals(globals) {}

  // Lower a print operation to a call to printf, puts, or putc
  LogicalResult
  matchAndRewrite(sim::PrintFormattedProcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    SmallString<32> fmtString;
    SmallVector<Value> arguments;
    arguments.reserve(op.getNumOperands());

    bool usePuts = false;
    bool usePutc = false;
    bool singleOperand = (op.getNumOperands() == 1);

    for (unsigned operandIdx = 0; operandIdx < op.getNumOperands();
         ++operandIdx) {
      auto defOp = op.getOperand(operandIdx).getDefiningOp();

      if (!isInlinableFormat(defOp)) {
        // Unsupported formattings are stringified seperately and passed as
        // char*
        fmtString += "%s";
        arguments.push_back(adaptor.getInputs()[operandIdx]);
        if (singleOperand)
          usePuts = true;
        continue;
      }

      TypeSwitch<Operation *>(defOp)
          .Case<sim::FormatLitOp>([&](sim::FormatLitOp fmtOp) {
            if (singleOperand) {
              usePutc = fmtOp.getLiteral().size() == 1;
              usePuts = !usePutc;
            }
            // Escape the string as required
            for (auto c : fmtOp.getLiteral())
              if (c == '%' && !usePuts && !usePutc)
                fmtString += "%%";
              else
                fmtString.push_back(c);
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
                auto shiftCst = rewriter.createOrFold<LLVM::ConstantOp>(
                    op.getLoc(), fmtOp.getValue().getType(),
                    remainingWidth - chunkWidth);
                argValue = rewriter.createOrFold<LLVM::LShrOp>(
                    op.getLoc(), fmtOp.getValue(), shiftCst);
              }

              if (width < 64)
                argValue = rewriter.createOrFold<LLVM::ZExtOp>(
                    op.getLoc(), rewriter.getI64Type(), argValue);
              else if (width > 64)
                argValue = rewriter.createOrFold<LLVM::TruncOp>(
                    op.getLoc(), rewriter.getI64Type(), argValue);

              arguments.push_back(argValue);
              remainingWidth -= chunkWidth;
              chunkWidth = 64;
            }
          })
          .Case<sim::FormatDecOp>([&](sim::FormatDecOp fmtOp) {
            Value argVal = fmtOp.getValue();
            auto width = fmtOp.getValue().getType().getIntOrFloatBitWidth();

            auto widthOp = rewriter.createOrFold<LLVM::ConstantOp>(
                op.getLoc(), rewriter.getI32Type(),
                sim::FormatDecOp::getDecimalWidth(width, fmtOp.getIsSigned()));
            arguments.push_back(widthOp);

            if (fmtOp.getIsSigned()) {
              fmtString += "%*lld";
              if (width < 64)
                argVal = rewriter.createOrFold<LLVM::SExtOp>(
                    op.getLoc(), rewriter.getI64Type(), fmtOp.getValue());
            } else {
              fmtString += "%*llu";
              if (width < 64)
                argVal = rewriter.createOrFold<LLVM::ZExtOp>(
                    op.getLoc(), rewriter.getI64Type(), fmtOp.getValue());
            }
            arguments.push_back(argVal);
          })
          .Case<sim::FormatBinOp>([&](sim::FormatBinOp fmtOp) {
            assert(fmtOp.getValue().getType() == rewriter.getI1Type());
            // Inline single bit binary substitutions as character.
            if (singleOperand)
              usePutc = true;
            auto cst0 = rewriter.createOrFold<LLVM::ConstantOp>(
                op.getLoc(), rewriter.getI32Type(), '0');
            auto cst1 = rewriter.createOrFold<LLVM::ConstantOp>(
                op.getLoc(), rewriter.getI32Type(), '1');
            auto selOp = rewriter.createOrFold<LLVM::SelectOp>(
                op.getLoc(), fmtOp.getValue(), cst1, cst0);
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
              valOp = rewriter.createOrFold<LLVM::TruncOp>(
                  op.getLoc(), rewriter.getI8Type(), fmtOp.getValue());
            }
            auto extOp = rewriter.createOrFold<LLVM::ZExtOp>(
                op.getLoc(), rewriter.getI32Type(), valOp);
            arguments.push_back(extOp);
          })
          .Default([&](auto fmtOp) {
            op.emitError("Unable to inline format string.");
          });
    }

    assert(!(usePutc && usePuts));

    // Lookup or create the function symbol.
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto int32Ty = rewriter.getI32Type();

    LLVM::LLVMFunctionType printFnType;
    StringRef printFnName;
    if (usePutc) {
      printFnType = LLVM::LLVMFunctionType::get(int32Ty, {int32Ty, ptrTy});
      printFnName = arcStlSymFputc;
    } else if (usePuts) {
      printFnType = LLVM::LLVMFunctionType::get(int32Ty, {ptrTy, ptrTy});
      printFnName = arcStlSymFputs;
    } else {
      printFnType = LLVM::LLVMFunctionType::get(int32Ty, {ptrTy, ptrTy}, true);
      printFnName = arcStlSymFprintf;
    }

    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return op.emitError("Unable to find parent module.");

    auto printFuncOp = lookupOrInsertExternalFunction(rewriter, moduleOp,
                                                      printFnName, printFnType);

    // Insert the format string if not already available.
    Value formatStrGlobalPtr;
    if (!usePutc) {
      fmtString.push_back('\0');

      auto symName =
          globals.newName(usePuts ? "_arc_puts_str" : "_arc_printf_fmtstr");
      LLVM::GlobalOp formatStrGlobal;
      if (!(formatStrGlobal = moduleOp.lookupSymbol<LLVM::GlobalOp>(symName))) {
        ConversionPatternRewriter::InsertionGuard insertGuard(rewriter);

        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto globalType =
            LLVM::LLVMArrayType::get(rewriter.getI8Type(), fmtString.size());
        formatStrGlobal = rewriter.create<LLVM::GlobalOp>(
            op.getLoc(), globalType, /*isConstant=*/true,
            LLVM::Linkage::Internal,
            /*name=*/symName, rewriter.getStringAttr(fmtString),
            /*alignment=*/0);
      }

      formatStrGlobalPtr =
          rewriter.create<LLVM::AddressOfOp>(op.getLoc(), formatStrGlobal);
    }

    // Call a function to retrieve the file stream to print to.
    auto streamFn = lookupOrInsertExternalFunction(
        rewriter, moduleOp, arcEnvSymGetPrintStream,
        LLVM::LLVMFunctionType::get(ptrTy, {int32Ty}));
    auto cst0 = rewriter.create<LLVM::ConstantOp>(op.getLoc(), int32Ty, 0);
    auto streamFnCall =
        rewriter.create<LLVM::CallOp>(op.getLoc(), streamFn, ValueRange{cst0});

    SmallVector<Value> callArgs;
    if (usePutc) {
      if (arguments.empty()) {
        // Constant fputc
        auto charCst = rewriter.createOrFold<LLVM::ConstantOp>(
            op.getLoc(), int32Ty, fmtString[0]);
        callArgs.push_back(charCst);
        callArgs.push_back(streamFnCall.getResult());
      } else {
        // Dynamic fputc
        callArgs.push_back(arguments.front());
        callArgs.push_back(streamFnCall.getResult());
      }
    } else if (usePuts) {
      if (arguments.empty()) {
        // Constant fputs
        callArgs.push_back(formatStrGlobalPtr);
        callArgs.push_back(streamFnCall.getResult());
      } else {
        // Dynamic fputs
        callArgs.push_back(arguments.front());
        callArgs.push_back(streamFnCall.getResult());
      }
    } else {
      // fprintf
      callArgs.push_back(streamFnCall.getResult());
      callArgs.push_back(formatStrGlobalPtr);
      callArgs.append(arguments);
    }

    // Finally, do the call to the print function. We don't care about the
    // return value right now.
    rewriter.create<LLVM::CallOp>(op.getLoc(), printFuncOp, callArgs);
    rewriter.eraseOp(op);

    return success();
  }

private:
  Namespace &globals;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerArcToLLVMPass : public LowerArcToLLVMBase<LowerArcToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void LowerArcToLLVMPass::runOnOperation() {
  // Collect the symbols in the root op such that the HW-to-LLVM lowering can
  // create LLVM globals with non-colliding names.
  Namespace globals;
  SymbolCache cache;
  cache.addDefinitions(getOperation());
  globals.add(cache);

  globals.newName(arcEnvSymGetPrintStream);
  globals.newName(arcStlSymFprintf);
  globals.newName(arcStlSymFputs);
  globals.newName(arcStlSymFputc);

  // Setup the conversion target. Explicitly mark `scf.yield` legal since it
  // does not have a conversion itself, which would cause it to fail
  // legalization and for the conversion to abort. (It relies on its parent op's
  // conversion to remove it.)
  LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<scf::YieldOp>(); // quirk of SCF dialect conversion

  // Inlined formatting tokens become trivially dead after lowering of
  // all print operations.
  target.addLegalOp<sim::FormatLitOp, sim::FormatHexOp, sim::FormatCharOp>();
  // Non-inlined tokens need to be converted to pointers
  // to their char buffers.
  target.addDynamicallyLegalOp<sim::FormatBinOp, sim::FormatDecOp>(
      isInlinableFormat);

  // Setup the arc dialect type conversion.
  LLVMTypeConverter converter(&getContext());
  converter.addConversion([&](seq::ClockType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([&](sim::FormatStringType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](StorageType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](MemoryType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](StateType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](SimModelInstanceType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  // Setup the conversion patterns.
  RewritePatternSet patterns(&getContext());

  // MLIR patterns.
  populateSCFToControlFlowConversionPatterns(patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);

  // CIRCT patterns.
  DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp> constAggregateGlobalsMap;
  populateHWToLLVMConversionPatterns(converter, patterns, globals,
                                     constAggregateGlobalsMap);
  populateHWToLLVMTypeConversions(converter);
  populateCombToLLVMConversionPatterns(converter, patterns);

  // Arc patterns.
  // clang-format off
  patterns.add<
    AllocMemoryOpLowering,
    AllocStateLikeOpLowering<arc::AllocStateOp>,
    AllocStateLikeOpLowering<arc::RootInputOp>,
    AllocStateLikeOpLowering<arc::RootOutputOp>,
    AllocStorageOpLowering,
    ClockGateOpLowering,
    FormatBinOpLowering,
    FormatDecOpLowering,
    MemoryReadOpLowering,
    MemoryWriteOpLowering,
    ModelOpLowering,
    ReplaceOpWithInputPattern<seq::ToClockOp>,
    ReplaceOpWithInputPattern<seq::FromClockOp>,
    SeqConstClockLowering,
    SimEmitValueOpLowering,
    StateReadOpLowering,
    StateWriteOpLowering,
    StorageGetOpLowering,
    ZeroCountOpLowering
  >(converter, &getContext());
  // clang-format on

  patterns.add<PrintFormattedOpLowering>(converter, &getContext(), globals);

  SmallVector<ModelInfo> models;
  if (failed(collectModels(getOperation(), models))) {
    signalPassFailure();
    return;
  }

  llvm::DenseMap<StringRef, ModelInfoMap> modelMap(models.size());
  for (ModelInfo &modelInfo : models) {
    llvm::DenseMap<StringRef, StateInfo> states(modelInfo.states.size());
    for (StateInfo &stateInfo : modelInfo.states)
      states.insert({stateInfo.name, stateInfo});
    modelMap.insert({modelInfo.name,
                     ModelInfoMap{modelInfo.numStateBytes, std::move(states)}});
  }

  patterns.add<SimInstantiateOpLowering, SimSetInputOpLowering,
               SimGetPortOpLowering, SimStepOpLowering>(
      converter, &getContext(), modelMap);

  // Apply the conversion.
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> circt::createLowerArcToLLVMPass() {
  return std::make_unique<LowerArcToLLVMPass>();
}
