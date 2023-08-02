#ifndef CIRCT_DIALECT_HW_APLOGIC_H
#define CIRCT_DIALECT_HW_APLOGIC_H

#include "HWLogicCode.h"

#include "circt/Dialect/HW/HWAPLogicConstant.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <utility>

namespace circt {
namespace hw {

class APLogic {
public:
  typedef APLogicConstant::LOGIC_FULL LOGIC[4];

  /// Returns the number of logic digits in the APLogic value
  unsigned getBitWidth() const { return BitWidth; };

  /// Construct zero-filled
  APLogic(const unsigned width) : BitWidth(width) {
    assert(BitWidth > 0 && "zero length logic values not allowed");
    if (isSelfContained()) {
      memset(&VAL.l9, 0, sizeof(LOGIC));
    } else {
      auto nSegments = getSegmentCount();
      VAL.ptr = new LOGIC[nSegments];
      memset(VAL.ptr, 0, sizeof(LOGIC) * nSegments);
    }
  }

  /// Construct initialized with unsigned integer
  APLogic(const unsigned width, APLogicConstant::LOGIC_FULL init)
      : BitWidth(width) {
    assert(BitWidth > 0 && "zero length logic values not allowed");
    if (isSelfContained()) {
      VAL.l9[0] = init;
      VAL.l9[1] = 0;
      VAL.l9[2] = 0;
      VAL.l9[3] = 0;
    } else {
      auto nSegments = getSegmentCount();
      memset(VAL.ptr, 0, sizeof(LOGIC) * nSegments);
      VAL.ptr[0][0] = init;
    }
  }

  /// Construct from string with an optional minimum width. Extra digits will be
  /// zero-filled.
  APLogic(const std::string initStr, unsigned minWidth = 0);

  /// Copy constructor
  APLogic(const APLogic &apl) : BitWidth(apl.BitWidth) {
    if (isSelfContained()) {
      memcpy(&VAL.l9, &apl.VAL.l9, sizeof(LOGIC));
    } else {
      auto nSegments = getSegmentCount();
      VAL.ptr = new LOGIC[nSegments];
      memcpy(VAL.ptr, apl.VAL.ptr, sizeof(LOGIC) * nSegments);
    }
  }

  /// Move constructor
  APLogic(APLogic &&apl) : BitWidth(apl.BitWidth) {
    memcpy(&VAL, &apl.VAL, sizeof(VAL));
    apl.invalidate();
  }

  /// Construct from APLogicConstant with an optional minimum width. Extra
  /// digits will be zero-filled.
  APLogic(const APLogicConstant &apc, unsigned minWidth = 0)
      : BitWidth(std::max(apc.BitWidth, minWidth)) {
    if (apc.isSelfContained() && isSelfContained()) {
      switch (apc.codeLength) {
      case LogicCodeLength::Log2:
        logcode::copy(VAL.l9, apc.VAL.l2);
        return;
      case LogicCodeLength::Log4:
        logcode::copy(VAL.l9, apc.VAL.l4);
        return;
      default:
        logcode::copy(VAL.l9, apc.VAL.l9);
        return;
      }
    }
    constructFromConstantSlowCase(apc);
  }

  /// Destructor
  ~APLogic() {
    if (!isSelfContained()) {
      delete[] VAL.ptr;
    }
  }

  friend void swap(APLogic &apl1, APLogic &apl2) {
    std::swap(apl1.BitWidth, apl2.BitWidth);
    std::swap(apl1.VAL, apl2.VAL);
  }

  /// Copy assignment operator
  APLogic &operator=(const APLogic &RHS) {
    if (this == &RHS)
      return *this;

    // Avoid reallocation if possible
    auto nSegments = getSegmentCount();
    if (nSegments == RHS.getSegmentCount()) {
      BitWidth = RHS.BitWidth;
      memcpy(getValuePointer(), RHS.getValuePointer(),
             sizeof(LOGIC) * nSegments);
    } else {
      auto copy = APLogic(RHS);
      swap(*this, copy);
    }

    return *this;
  }

  /// Move assignment operator
  APLogic &operator=(APLogic &&that) {
#ifdef EXPENSIVE_CHECKS
    // Some std::shuffle implementations still do self-assignment.
    if (this == &that)
      return *this;
#endif
    assert(this != &that && "Self-move not supported");
    if (!isSelfContained())
      delete[] VAL.ptr;

    // Use memcpy so that type based alias analysis sees both VAL and pVal
    // as modified.
    memcpy(&VAL, &that.VAL, sizeof(VAL));

    BitWidth = that.BitWidth;
    that.invalidate();
    return *this;
  }

  /// Returns the minimum code length required to represent the value currently
  /// contained in this APLogic
  LogicCodeLength getMinimumRequiredCode() const;

  void operator^=(const APLogic &that) {
    assert(BitWidth == that.BitWidth &&
           "APLogic operations require operands of same width");
    auto thisLogPtr = getValuePointer();
    auto thatLogPtr = that.getValuePointer();
    for (size_t i = 0; i < getSegmentCount(); i++)
      logcode::opXor_inplace(thisLogPtr[i], thatLogPtr[i]);
  }

  void operator^=(const APLogicConstant &that) {
    if (that.isSelfContained()) {
      assert(isSelfContained());
      switch (that.codeLength) {
      case LogicCodeLength::Log2:
        logcode::opXor_inplace(VAL.l9, that.VAL.l2);
        return;
      case LogicCodeLength::Log4:
        logcode::opXor_inplace(VAL.l9, that.VAL.l4);
        return;
      default:
        logcode::opXor_inplace(VAL.l9, that.VAL.l9);
        return;
      }
    }
    xorConstSlowCase(that);
  }

  void operator<<=(unsigned shamt);
  void rShiftS(unsigned shamt);

  /// Replace the digits stating at 'index' with 'length' digits starting at
  /// 'srcOffset' in 'apl' repeated 'repeat'-times. Source and destination may
  /// overlap. This operation may not exceed the width of the current instance.
  void replace(const APLogic &apl, unsigned index, unsigned length,
               unsigned repeat, unsigned srcOffset) {
    // Operate on a copy if there is a (partial) overlap between destination and
    // source digits
    bool isOverlapping = false;
    if (&apl == this) {
      if (index >= srcOffset)
        isOverlapping = srcOffset + length > index;
      else
        isOverlapping = index + repeat * length > srcOffset;
    }

    if (isOverlapping) {
      replaceNonOverlapping(APLogic(apl), index, length, repeat, srcOffset);
    } else {
      replaceNonOverlapping(apl, index, length, repeat, srcOffset);
    }
  }

  /// Replace the digits stating at 'index' with the given APLogic.
  /// This operation may not exceed the width of the current instance.
  void replace(const APLogic &apl, unsigned index) {
    replace(apl, index, apl.BitWidth, 1, 0);
  };

  /// Replace the digits stating at 'index' with 'length' digits starting at
  /// 'srcOffset' in 'apc' repeated 'repeat'-times. This operation may not
  /// exceed the width of the current instance.
  void replace(const APLogicConstant &apc, unsigned index, unsigned length,
               unsigned repeat, unsigned srcOffset) {
    if (repeat == 0 || length == 0)
      return;
    assert(srcOffset + length <= apc.BitWidth &&
           "length exceeds bounds of source APLogicConstant");
    assert(length * repeat + index <= BitWidth &&
           "length exceeds bounds of destination APLogic");
    if (isSelfContained() && apc.isSelfContained()) {
      if (apc.getCodeLength() == LogicCodeLength::Log2)
        logcode::copySlice(VAL.l9, apc.VAL.l2, srcOffset, index, length,
                           repeat);
      else if (apc.getCodeLength() == LogicCodeLength::Log4)
        logcode::copySlice(VAL.l9, apc.VAL.l4, srcOffset, index, length,
                           repeat);
      else
        logcode::copySlice(VAL.l9, apc.VAL.l9, srcOffset, index, length,
                           repeat);
      return;
    }
    replaceNonOverlapping(APLogic(apc), index, length, repeat, srcOffset);
  };

  /// Replace the digits stating at 'index' with the given APLogicConstant.
  /// This operation may not exceed the width of the current instance.
  void replace(const APLogicConstant &apc, unsigned index) {
    replace(apc, index, apc.BitWidth, 1, 0);
  }

  void setDigitAtIndex(logcode::LogDigit digit, unsigned index);
  logcode::LogDigit getDigitAtIndex(unsigned index) const;

  std::string toString() const;

private:
  friend class APLogicConstant;
  union {
    LOGIC l9;
    LOGIC *ptr;
  } VAL;

  unsigned BitWidth;

  static constexpr LogicCodeLength codeLength = LogicCodeLength::Log9;

  static constexpr bool isSelfContained(unsigned BitWidth) {
    return BitWidth <= maxDigitsByCode<LOGIC>(codeLength);
  }

  constexpr size_t getFieldSize() const { return sizeof(LOGIC) / 4; }

  bool isSelfContained() const { return isSelfContained(BitWidth); }

  size_t getSegmentCount() const {
    return APLogicConstant::getSegmentCount(BitWidth, getFieldSize());
  }

  LOGIC *getValuePointer() { return isSelfContained() ? &VAL.l9 : VAL.ptr; };
  const LOGIC *getValuePointer() const {
    return isSelfContained() ? &VAL.l9 : VAL.ptr;
  };

  void invalidate() { BitWidth = 0; }

  void constructFromConstantSlowCase(const APLogicConstant &apc);
  void xorConstSlowCase(const APLogicConstant &that);

  void replaceNonOverlapping(const APLogic &apl, unsigned index,
                             unsigned length, unsigned repeat,
                             unsigned srcOffset);

  void replaceNonOverlapping(const APLogic &apl, unsigned index) {
    replaceNonOverlapping(apl, index, apl.BitWidth, 1, 0);
  };
};

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_APLOGIC_H
