#include "circt/Dialect/HW/HWAPLogic.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

using namespace circt::hw;
using namespace llvm;

LogicCodeLength APLogic::getMinimumRequiredCode() const {
  bool isL2 = true;
  bool isL4 = true;

  const LOGIC *logptr = getValuePointer();
  size_t count = getSegmentCount();

  for (size_t i = 0; i < count; i++) {
    if (logcode::getNon01XZMask(logptr[i]) != 0) {
      isL2 = false;
      isL4 = false;
      break;
    }
    if (logcode::getUnknownMask(logptr[i]) != 0) {
      isL2 = false;
    }
  }

  if (!isL2 && !isL4)
    return LogicCodeLength::Log9;
  if (!isL2)
    return LogicCodeLength::Log4;
  return LogicCodeLength::Log2;
}

void APLogic::setDigitAtIndex(logcode::LogDigit digit, unsigned index) {
  assert(index < BitWidth && "APLogic index out of bounds");
  assert(logcode::isValidLogicDigit(digit) && "invalid logic digit");
  const unsigned bitPos = index % maxDigitsByCode<LOGIC>(codeLength);
  const unsigned segmentPos = index / maxDigitsByCode<LOGIC>(codeLength);
  logcode::fillWithValue(getValuePointer()[segmentPos], digit,
                         ((APLogicConstant::LOGIC_FULL)1) << bitPos);
}

logcode::LogDigit APLogic::getDigitAtIndex(unsigned index) const {
  assert(index < BitWidth && "APLogic index out of bounds");
  const unsigned bitPos = index % maxDigitsByCode<LOGIC>(codeLength);
  const unsigned segmentPos = index / maxDigitsByCode<LOGIC>(codeLength);
  return logcode::digitAtIndex(getValuePointer()[segmentPos], bitPos);
}

APLogic::APLogic(const std::string initStr, unsigned minWidth)
    : APLogic(std::max((unsigned)initStr.size(), minWidth)) {
  auto cString = initStr.c_str();
  const auto stringLength = initStr.size();
  bool success = true;
  if (isSelfContained()) {
    success = logcode::fromCStr(VAL.l9, cString, stringLength);
  } else {
    auto chunkLen = stringLength % maxDigitsByCode<LOGIC>(codeLength);
    if (chunkLen == 0)
      chunkLen = maxDigitsByCode<LOGIC>(codeLength);

    auto offset = 0;
    auto nSegments = getSegmentCount();
    auto paddingSegments = nSegments - APLogicConstant::getSegmentCount(
                                           stringLength, getFieldSize());
    if (paddingSegments > 0)
      memset(VAL.ptr[nSegments - paddingSegments], 0,
             sizeof(LOGIC) * paddingSegments);
    for (size_t i = paddingSegments; i < nSegments; i++) {
      success &= logcode::fromCStr(VAL.ptr[nSegments - i - 1], cString + offset,
                                   chunkLen);
      offset += chunkLen;
      chunkLen = maxDigitsByCode<LOGIC>(codeLength);
    }
  }
  assert(success && "Not a valid logic literal.");
}

void APLogic::constructFromConstantSlowCase(const APLogicConstant &apc) {

  auto destSegments = getSegmentCount();
  auto srcSegments = apc.getSegmentCount();
  assert(srcSegments <= destSegments);
  unsigned expansionSegments = destSegments - srcSegments;

  LOGIC *logPtr; // Pointer to our data
  if (isSelfContained()) {
    logPtr = &VAL.l9;
  } else {
    // Allocate space for more than 64 digits
    VAL.ptr = new LOGIC[destSegments];
    logPtr = VAL.ptr;
    if (apc.isSelfContained()) {
      // Expand compacted APLogicConstant into our first 64 digits
      switch (apc.codeLength) {
      case LogicCodeLength::Log2:
        logcode::copy(logPtr[0], apc.VAL.l2);
        break;
      case LogicCodeLength::Log4:
        logcode::copy(logPtr[0], apc.VAL.l4);
        break;
      default:
        logcode::copy(logPtr[0], apc.VAL.l9);
        break;
      }
    }
  }

  if (!apc.isSelfContained()) {
    // APLogicConstant has an allocation, so fields should be 64 bit wide
    assert(apc.getFieldSize() == getFieldSize());

    if (apc.codeLength == codeLength) {
      // L9 to L9, just copy the entire block
      memcpy(logPtr, apc.VAL.ptr, srcSegments * sizeof(LOGIC));
    } else {
      // Expand the logic code by interleaving zeros
      for (size_t i = 0; i < srcSegments; i++) {
        if (apc.codeLength == LogicCodeLength::Log2) {
          // L2 to L9
          logcode::copy(logPtr[i], apc.VAL.ptr_l2[i]);
        } else {
          // L4 to L9
          logcode::copy(logPtr[i], apc.VAL.ptr_l4[i]);
        }
      }
    }
  }

  if (expansionSegments > 0) {
    // Clear expansion
    memset(logPtr + srcSegments, 0, expansionSegments * sizeof(LOGIC));
  }
}

std::string APLogic::toString() const {
  std::string s(BitWidth, '\0');
  const logcode::L9_64 *logPtr = getValuePointer();
  size_t segment = getSegmentCount() - 1;
  for (unsigned i = 0; i < BitWidth; i++) {
    auto subIdx = (BitWidth - 1 - i) % maxDigitsByCode<LOGIC>(codeLength);
    auto digit = logcode::digitAtIndex(logPtr[segment], subIdx);
    s[i] = logcode::logDigitToChar(digit);
    if (subIdx == 0)
      segment--;
  }
  return s;
}

void APLogic::xorConstSlowCase(const APLogicConstant &that) {
  auto nSegments = getSegmentCount();
  auto *valPtr =
      reinterpret_cast<APLogicConstant::LOGIC_FULL *>(getValuePointer());

  assert(that.getSegmentCount() == nSegments);
  assert(that.getFieldSize() == getFieldSize());
  assert(!that.isSelfContained());

  const auto thisStride = (unsigned)codeLength;
  const auto thatStride = (unsigned)that.codeLength;

  for (size_t i = 0; i < nSegments; i++)
    logcode::opXor_unsafe(valPtr + i * thisStride, thisStride,
                          valPtr + i * thisStride, thisStride,
                          that.VAL.ptr + i * thatStride, thatStride);
}

void APLogic::replaceNonOverlapping(const APLogic &apl, unsigned index,
                                    unsigned length, unsigned repeat,
                                    unsigned srcOffset) {

  if (repeat == 0 || length == 0)
    return;

  assert(srcOffset + length <= apl.BitWidth &&
         "length exceeds bounds of source APLogicConstant");
  assert(length * repeat + index <= BitWidth &&
         "length exceeds bounds of destination APLogic");

  auto *valPtr = getValuePointer();
  auto *srcPtr = apl.getValuePointer();

  const auto fieldDigits = maxDigitsByCode<LOGIC>(codeLength);

  for (unsigned i = 0; i < repeat; i++) {
    unsigned repeatOffset = i * length;
    unsigned insertCount = 0;
    while (insertCount < length) {
      auto iterSrcOffset = (insertCount + srcOffset) % fieldDigits;
      auto srcSegment = (insertCount + srcOffset) / fieldDigits;
      auto iterDestOffset = (insertCount + index + repeatOffset) % fieldDigits;
      auto destSegment = (insertCount + index + repeatOffset) / fieldDigits;
      auto chunkWidth =
          std::min(fieldDigits - iterSrcOffset, fieldDigits - iterDestOffset);

      chunkWidth = std::min(chunkWidth, (size_t)length - insertCount);
      logcode::copySlice(valPtr[destSegment], srcPtr[srcSegment], iterSrcOffset,
                         iterDestOffset, chunkWidth);

      insertCount += chunkWidth;
    }
  }
}

void APLogic::operator<<=(unsigned shamt) {
  LOGIC *logPtr = getValuePointer();
  const auto nSegments = getSegmentCount();
  const auto fieldBits = 8 * getFieldSize();

  if (shamt >= BitWidth) {
    memset(logPtr, 0, nSegments * sizeof(LOGIC));
    return;
  }

  if (shamt >= fieldBits) {
    auto stride = shamt / fieldBits;
    memmove(logPtr + stride, logPtr, sizeof(LOGIC) * (nSegments - stride));
    memset(logPtr, 0, sizeof(LOGIC) * stride);
    shamt = shamt % fieldBits;
  }

  if (shamt == 0)
    return;

  for (size_t i = 0; i < nSegments; i++) {
    const auto segIdx = nSegments - i - 1;
    if (i != 0) {
      logcode::copySlice(logPtr[segIdx + 1], logPtr[segIdx], fieldBits - shamt,
                         0, shamt);
    }
    logcode::opShl(logPtr[segIdx], shamt);
  }

  APLogicConstant::LOGIC_FULL mask = (1UL << (BitWidth % fieldBits)) - 1;
  for (unsigned i = 0; i < 4; i++)
    logPtr[nSegments - 1][i] &= mask;
}

void APLogic::rShiftS(unsigned shamt) {
  LOGIC *logPtr = getValuePointer();
  const auto nSegments = getSegmentCount();
  const auto fieldBits = 8 * getFieldSize();

  if (shamt == 0)
    return;

  // sign extend msb
  auto msbBits = BitWidth % fieldBits;
  if (msbBits != 0) {
    logcode::opShl(logPtr[nSegments - 1], fieldBits - msbBits);
    logcode::opShrs(logPtr[nSegments - 1], fieldBits - msbBits);
  }

  LOGIC fillValue;
  logcode::copy(fillValue, logPtr[nSegments - 1]);
  logcode::opShrs(fillValue, 63);

  if (shamt >= fieldBits) {
    auto stride = shamt / fieldBits;
    memmove(logPtr, logPtr + stride, sizeof(LOGIC) * (nSegments - stride));
    for (size_t i = 0; i < stride; i++) {
      memcpy(logPtr + (nSegments - stride + i), &fillValue, sizeof(LOGIC));
    }
    shamt = shamt % fieldBits;
  }

  if (shamt != 0) {
    for (size_t i = 0; i < nSegments; i++) {
      logcode::opShru(logPtr[i], shamt);
      if (i != nSegments - 1) {
        logcode::copySlice(logPtr[i], logPtr[i + 1], 0, fieldBits - shamt,
                           shamt);
      }
    }
  }

  // Mask unused MSBs
  uint64_t mask = UINT64_MAX;
  mask >>= 64 - BitWidth;
  for (unsigned i = 0; i < 4; i++)
    logPtr[nSegments - 1][i] &= mask;
}
