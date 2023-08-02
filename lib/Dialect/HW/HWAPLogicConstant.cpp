#include "circt/Dialect/HW/HWAPLogicConstant.h"
#include "circt/Dialect/HW/HWAPLogic.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

using namespace circt::hw;
using namespace llvm;

APLogicConstant::APLogicConstant(const APLogic &apl)
    : BitWidth(apl.getBitWidth()), codeLength(apl.getMinimumRequiredCode()) {
  assert(apl.codeLength == LogicCodeLength::Log9);
  if (isSelfContained()) {
    assert(apl.isSelfContained());
    assert(getFieldSize() <= apl.getFieldSize());

    if (codeLength == LogicCodeLength::Log2)
      logcode::copyAndCast(VAL.l2, apl.VAL.l9);
    else if (codeLength == LogicCodeLength::Log4)
      logcode::copyAndCast(VAL.l4, apl.VAL.l9);
    else
      logcode::copyAndCast(VAL.l9, apl.VAL.l9);
    return;
  }

  assert(getFieldSize() == apl.getFieldSize());
  const auto nSegments = getSegmentCount();
  VAL.ptr = new LOGIC_FULL[nSegments * (size_t)codeLength];

  if (codeLength == LogicCodeLength::Log9) {
    // We need L9, so just copy
    size_t nBytes =
        apl.getSegmentCount() * apl.getFieldSize() * (size_t)apl.codeLength;
    memcpy(VAL.ptr, apl.getValuePointer(), nBytes);
    return;
  }

  // Squash L9 to L4/L2
  auto destStride = (size_t)codeLength;
  const APLogic::LOGIC *sourcePtr = apl.getValuePointer();
  for (size_t i = 0; i < nSegments; i++) {
    memcpy(VAL.ptr + (i * destStride), sourcePtr + i,
           destStride * sizeof(LOGIC_FULL));
  }
}

llvm::APInt APLogicConstant::toAPIntSlowCase() const {
  assert(!isSelfContained() && codeLength == LogicCodeLength::Log9);
  assert(isIntegerLike());
  auto nSegments = getSegmentCount();

  llvm::SmallVector<uint64_t, 4> intValues;
  intValues.reserve(nSegments);
  for (size_t i = 0; i < nSegments; i++) {
    intValues.push_back(VAL.ptr_l9[i][0]);
  }

  return llvm::APInt(BitWidth, ArrayRef<uint64_t>(intValues));
}

bool APLogicConstant::containsUnknownValuesSlowCase() const {
  assert(!isSelfContained());
  assert(codeLength == LogicCodeLength::Log9);

  for (size_t i = 0; i < getSegmentCount(); i++) {
    if (logcode::getUnknownMask(VAL.ptr_l9[i]) != 0)
      return true;
  }
  return false;
}

bool APLogicConstant::isZeroLikeSlowCase() const {
  assert(codeLength == LogicCodeLength::Log9);

  if (isSelfContained()) {
    return (logcode::toInt(VAL.l9) == 0) &&
           (logcode::getUnknownMask(VAL.l9) == 0);
  }

  for (size_t i = 0; i < getSegmentCount(); i++) {
    if ((logcode::toInt(VAL.ptr_l9[i]) != 0) ||
        (logcode::getUnknownMask(VAL.ptr_l9[i]) != 0))
      return false;
  }

  return true;
}

APLogicConstant APLogicConstant::concatSlowCase(const APLogicConstant &msb,
                                                const APLogicConstant &lsb) {
  auto apCombined = APLogic(lsb, msb.BitWidth);
  apCombined.replace(APLogic(msb), lsb.BitWidth);
  return APLogicConstant(apCombined);
}

APLogicConstant APLogicConstant::lshiftSlowCase(unsigned shamt) const {
  auto apl = APLogic(*this);
  apl <<= shamt;
  return APLogicConstant(apl);
}

llvm::hash_code circt::hw::hash_value(const APLogicConstant &Arg) {
  if (Arg.isSelfContained())
    return hash_combine(Arg.BitWidth, (uint32_t)Arg.codeLength, Arg.VAL.raw);

  size_t length = (size_t)Arg.codeLength * Arg.getSegmentCount();

  return hash_combine(Arg.BitWidth, (uint32_t)Arg.codeLength,
                      hash_combine_range(Arg.VAL.ptr, Arg.VAL.ptr + length));
}
