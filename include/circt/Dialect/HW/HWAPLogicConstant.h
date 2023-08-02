#ifndef CIRCT_DIALECT_HW_APLOGICCONSTANT_H
#define CIRCT_DIALECT_HW_APLOGICCONSTANT_H

#include "HWLogicCode.h"

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

template <typename T, typename Enable>
struct DenseMapInfo;

enum class LogicCodeLength : uint32_t { Log2 = 1, Log4 = 2, Log9 = 4 };

template <typename ST>
static constexpr size_t maxDigitsByCode(const LogicCodeLength codelen) {
  switch (codelen) {
  case LogicCodeLength::Log2:
    return sizeof(ST) * 8;
  case LogicCodeLength::Log4:
    return sizeof(ST) * 4;
  default:
    return sizeof(ST) * 2;
  }
}

class APLogic;

class APLogicConstant {
public:
  typedef uint64_t LOGIC_FULL;
  typedef uint32_t LOGIC_HALF;
  typedef uint16_t LOGIC_QUARTER;

  static_assert(std::is_same<LOGIC_FULL, llvm::APInt::WordType>::value);

  /// Returns an zero-filled APLogicConstant of the given width
  static APLogicConstant getAllZeros(unsigned int width) {
    return APLogicConstant(width, LogicCodeLength::Log2);
  }

  /// Returns an one-filled APLogicConstant of the given width
  static APLogicConstant getAllOnes(unsigned int width) {
    const auto maxDigits = maxDigitsByCode<LOGIC_FULL>(LogicCodeLength::Log2);
    if (width <= maxDigits)
      return APLogicConstant(LogicCodeLength::Log2, getOnesMask(width), width);
    const bool ones[1] = {true};
    return APLogicConstant(width, ones);
  }

  /// Construct zero-filled with given width and code length
  APLogicConstant(unsigned width, LogicCodeLength codeLen)
      : BitWidth(width), codeLength(codeLen) {
    assert(width > 0 && "zero length logic values not allowed");
    if (isSelfContained()) {
      VAL.raw = 0;
    } else {
      VAL.ptr = new LOGIC_FULL[getSegmentCount() * (size_t)codeLength];
      memset(VAL.ptr, 0,
             getSegmentCount() * (size_t)codeLength * sizeof(LOGIC_FULL));
    }
  }

  /// Construct from uint with given width
  /// Bits exceeding the specified width are trimmed. Additional bits are
  /// zero-filled.
  APLogicConstant(LOGIC_FULL uintValue, unsigned width)
      : APLogicConstant(width, LogicCodeLength::Log2) {
    const auto maxDigits = maxDigitsByCode<LOGIC_FULL>(LogicCodeLength::Log2);
    if (width < maxDigits) {
      uintValue &= getOnesMask(width);
    }
    if (isSelfContained())
      VAL.l2[0] = uintValue;
    else
      (*VAL.ptr_l2)[0] = uintValue;
  }

  // Construct from APInt
  APLogicConstant(const ::llvm::APInt &apint)
      : BitWidth(apint.getBitWidth()), codeLength(LogicCodeLength::Log2) {
    if (isSelfContained()) {
      assert(apint.isSingleWord());
      VAL.raw = apint.getZExtValue();
    } else {
      auto nSegments = getSegmentCount();
      assert(nSegments == apint.getNumWords());
      VAL.ptr = new LOGIC_FULL[nSegments];
      memcpy(VAL.ptr, apint.getRawData(), sizeof(LOGIC_FULL) * nSegments);
    }
  }

  /// Destructor
  ~APLogicConstant() {
    if (!isSelfContained())
      delete[] VAL.ptr;
  }

  // Get filled
  static APLogicConstant getFilled(logcode::LogDigit digit,
                                   unsigned int width) {
    assert(logcode::isValidLogicDigit(digit) && "invalid digit");
    auto codeLen = logcode::getCodeLengthForLogDigit(digit);
    if (codeLen == (unsigned)LogicCodeLength::Log2) {
      return (digit == logcode::LogDigit::LOGD_0) ? getAllZeros(width)
                                                  : getAllOnes(width);
    } else if (codeLen == (unsigned)LogicCodeLength::Log4) {
      const bool bVal[2] = {!!(digit & 1), !!(digit & 2)};
      return APLogicConstant(width, bVal);
    } else {
      const bool bVal[4] = {!!(digit & 1), !!(digit & 2), !!(digit & 4),
                            !!(digit & 8)};
      return APLogicConstant(width, bVal);
    }
  }

  template <char C>
  typename std::enable_if<
      logcode::isValidLogicDigit(logcode::charToLogDigit(C)),
      APLogicConstant>::type static getFilled(unsigned int width) {
    return getFilled(logcode::charToLogDigit(C), width);
  }

  /// Deleted default constructor
  APLogicConstant() = delete;

  /// Deleted assignment operator
  APLogicConstant &operator=(const APLogicConstant &that) = delete;

  /// Construct from APLogic
  APLogicConstant(const APLogic &apl);

  /// Copy constructor
  APLogicConstant(const APLogicConstant &apl)
      : BitWidth(apl.BitWidth), codeLength(apl.codeLength) {
    if (isSelfContained()) {
      VAL.raw = apl.VAL.raw;
    } else {
      VAL.ptr = new LOGIC_FULL[getSegmentCount() * (size_t)codeLength];
      memcpy(VAL.ptr, apl.VAL.ptr,
             getSegmentCount() * (size_t)codeLength * sizeof(LOGIC_FULL));
    }
  }

  /// Move constructor
  APLogicConstant(APLogicConstant &&apl)
      : BitWidth(apl.BitWidth), codeLength(apl.codeLength) {
    VAL.raw = apl.VAL.raw;
    apl.VAL.ptr = nullptr;
  }

  /// Returns the numer of bytes used to store the logic value
  size_t getSizeInBytes() const {
    return getFieldSize() * getSegmentCount() * (size_t)codeLength;
  }

  /// Returns true iff all contained logic digits are zero
  bool isZero() const {
    if (codeLength != LogicCodeLength::Log2)
      return false;
    if (isSelfContained())
      return VAL.l2[0] == 0;
    for (size_t i = 0; i < getSegmentCount(); i++) {
      if (VAL.ptr[i] != 0)
        return false;
    }
    return true;
  }

  /// Returns true iff all contained logic digits are zero or zero-like (i.e.
  /// 'L')
  bool isZeroLike() const {
    if (codeLength == LogicCodeLength::Log2)
      return isZero();
    if (codeLength == LogicCodeLength::Log4)
      return false;
    return isZeroLikeSlowCase();
  }

  /// Returns true iff the contained value is an integer, (i.e. only 0/1 digits)
  bool isInteger() const { return codeLength == LogicCodeLength::Log2; }

  /// Returns true iff the contained value is integer-like, (i.e. only 0/L/1/H
  /// digits)
  bool isIntegerLike() const { return !containsUnknownValues(); }

  /// Returns true iff the contained value contains alt least one digit
  /// representing an unknwown state
  bool containsUnknownValues() const {
    if (codeLength == LogicCodeLength::Log2)
      return false;
    if (codeLength == LogicCodeLength::Log4)
      return true;
    if (isSelfContained())
      return logcode::getUnknownMask(VAL.l9) != 0;
    return containsUnknownValuesSlowCase();
  }

  /// Returns the contained value as APInt.
  /// Triggers an assertion if the contained value is not an integer.
  llvm::APInt asAPInt() const {
    assert(isInteger());
    llvm::ArrayRef<LOGIC_FULL> intArray(getValuePointer(), getSegmentCount());
    return llvm::APInt(BitWidth, intArray);
  }

  /// Returns the contained value as APInt.
  /// Triggers an assertion if the contained value is not integer-like.
  llvm::APInt toAPInt() const {
    if (isInteger())
      return asAPInt();
    assert(isIntegerLike());
    // Log4 is never integer like
    assert(codeLength == LogicCodeLength::Log9);
    if (isSelfContained())
      return llvm::APInt(BitWidth, (LOGIC_FULL)logcode::asInt(VAL.l9));
    return toAPIntSlowCase();
  }

  /// Equality operator
  /// Two APLogicConstants are equal if they contain the same value and have the
  /// same width.
  bool operator==(const APLogicConstant &RHS) const {
    if (codeLength != RHS.codeLength)
      return false;
    if (BitWidth != RHS.BitWidth)
      return false;
    if (isSelfContained())
      return (VAL.raw == RHS.VAL.raw);
    return (memcmp(VAL.ptr, RHS.VAL.ptr, getSizeInBytes()) == 0);
  }

  APLogicConstant operator<<(unsigned shamt) const {
    if (shamt >= BitWidth)
      return APLogicConstant::getAllZeros(BitWidth);
    if (isInteger() && isSelfContained())
      return APLogicConstant(VAL.raw << shamt, BitWidth);
    return lshiftSlowCase(shamt);
  }

  /// Create a single-digit APLogicConstant from a character value
  template <char C>
  static constexpr APLogicConstant fromChar() {
    return fromDigit<logcode::charToLogDigit(C)>();
  }

  /// Returns the number of logic digits in the APLogic value
  unsigned getBitWidth() const { return BitWidth; };

  /// Returns the code length used to encode the contained logic value
  LogicCodeLength getCodeLength() const { return codeLength; };

  static APLogicConstant concat(const APLogicConstant &msb,
                                const APLogicConstant &lsb) {
    if (!msb.isSelfContained() || !lsb.isSelfContained())
      return concatSlowCase(msb, lsb);

    if (msb.codeLength != lsb.codeLength)
      if ((msb.codeLength != LogicCodeLength::Log2) &&
          (lsb.codeLength != LogicCodeLength::Log2))
        return concatSlowCase(msb, lsb);

    auto combinedWidth = msb.BitWidth + lsb.BitWidth;
    auto codeLength = std::max(msb.codeLength, lsb.codeLength);
    if (!isSelfContained(codeLength, combinedWidth))
      return concatSlowCase(msb, lsb);

    LOGIC_FULL newRaw = (msb.VAL.raw << lsb.BitWidth) | lsb.VAL.raw;
    return APLogicConstant(codeLength, newRaw, combinedWidth);
  }

  // Helpers for storing an APLogicConstant in attributes
  friend struct llvm::DenseMapInfo<APLogicConstant, void>;
  friend llvm::hash_code hash_value(const APLogicConstant &Arg);

private:
  friend class APLogic;

  typedef LOGIC_FULL L2TYPE[1];
  typedef LOGIC_HALF L4TYPE[2];
  typedef LOGIC_QUARTER L9TYPE[4];
  union {
    LOGIC_FULL raw;
    L9TYPE l9;
    L4TYPE l4;
    L2TYPE l2;

    LOGIC_FULL *ptr;
    LOGIC_FULL (*ptr_l9)[4];
    LOGIC_FULL (*ptr_l4)[2];
    LOGIC_FULL (*ptr_l2)[1];
  } VAL;

  const unsigned BitWidth;
  const LogicCodeLength codeLength;

  static constexpr LOGIC_FULL getOnesMask(unsigned n) {
    if (n >= sizeof(LOGIC_FULL) * 8)
      return ~((LOGIC_FULL)0);
    else
      return (((LOGIC_FULL)1) << n) - 1;
  }

  template <LogicCodeLength CL, unsigned W>
  static constexpr APLogicConstant fromRaw(LOGIC_FULL raw) {
    static_assert(W != 0);
    static_assert(isSelfContained(CL, W));
    return APLogicConstant(CL, raw, W);
  }

  constexpr APLogicConstant(LogicCodeLength _codeLength, LOGIC_FULL raw,
                            unsigned width)
      : VAL({raw}), BitWidth(width), codeLength(_codeLength){};

  // Construct filled with a given digit
  template <unsigned N, typename T>
  APLogicConstant(
      unsigned width, const T (&setFlag)[N],
      typename std::enable_if<N == 1 || N == 2 || N == 4>::type * = 0)
      : APLogicConstant(width, (LogicCodeLength)N) {
    assert(width > 0 && "zero length logic values not allowed");
    LOGIC_FULL masks[N];
    unsigned msbWidth = BitWidth % (sizeof(LOGIC_FULL) * 8);
    if (msbWidth == 0)
      msbWidth = sizeof(LOGIC_FULL) * 8;
    for (unsigned i = 0; i < N; i++) {
      if (!!setFlag[i]) {
        masks[i] = getOnesMask(msbWidth);
      } else {
        masks[i] = 0;
      }
    }
    if (isSelfContained()) {
      if (N == 4)
        logcode::copyAndCast(VAL.l9, masks);
      else if (N == 2)
        logcode::copyAndCast(VAL.l4, masks);
      else
        logcode::copyAndCast(VAL.l2, masks);
    } else {
      auto nSegments = getSegmentCount();
      for (size_t i = 0; i < nSegments * N; i++) {
        if (i >= ((nSegments - 1) * N))
          VAL.ptr[i] = masks[i % N];
        else
          VAL.ptr[i] = (!!setFlag[i % N]) ? ~((LOGIC_FULL)0) : 0;
      }
    }
  }

  static constexpr bool isSelfContained(LogicCodeLength codeLength,
                                        unsigned BitWidth) {
    if ((codeLength == LogicCodeLength::Log2) &&
        (BitWidth <= maxDigitsByCode<LOGIC_FULL>(LogicCodeLength::Log2)))
      return true;
    if ((codeLength == LogicCodeLength::Log4) &&
        (BitWidth <= maxDigitsByCode<LOGIC_FULL>(LogicCodeLength::Log4)))
      return true;
    if ((codeLength == LogicCodeLength::Log9) &&
        (BitWidth <= maxDigitsByCode<LOGIC_FULL>(LogicCodeLength::Log9)))
      return true;
    return false;
  }

  bool isSelfContained() const { return isSelfContained(codeLength, BitWidth); }

  static size_t getFieldSize(LogicCodeLength codeLength, unsigned BitWidth) {
    if ((codeLength == LogicCodeLength::Log9) &&
        (BitWidth <= maxDigitsByCode<LOGIC_FULL>(LogicCodeLength::Log9)))
      return sizeof(LOGIC_QUARTER);
    if ((codeLength == LogicCodeLength::Log4) &&
        (BitWidth <= maxDigitsByCode<LOGIC_FULL>(LogicCodeLength::Log4)))
      return sizeof(LOGIC_HALF);
    return sizeof(LOGIC_FULL);
  }

  size_t getFieldSize() const { return getFieldSize(codeLength, BitWidth); }

  static constexpr size_t getSegmentCount(unsigned BitWidth, size_t fieldSize) {
    auto fieldSizeBits = fieldSize * 8;
    return (BitWidth + (fieldSizeBits - 1)) / fieldSizeBits;
  }

  size_t getSegmentCount() const {
    return getSegmentCount(BitWidth, getFieldSize());
  }

  template <logcode::LogDigit LD>
  static constexpr APLogicConstant fromDigit() {
    const auto cl = (LogicCodeLength)logcode::getCodeLengthForLogDigit<LD>();
    union {
      LOGIC_FULL raw;
      L9TYPE l9;
      L4TYPE l4;
      L2TYPE l2;
    } newVal;
    const uint8_t digitVal = (uint8_t)LD;
    if (cl == LogicCodeLength::Log2) {
      newVal.l2[0] = (digitVal & 1) ? 1 : 0;
      return fromRaw<LogicCodeLength::Log2, 1>(newVal.raw);
    } else if (cl == LogicCodeLength::Log4) {
      newVal.l4[0] = (digitVal & 1) ? 1 : 0;
      newVal.l4[1] = (digitVal & 2) ? 1 : 0;
      return fromRaw<LogicCodeLength::Log4, 1>(newVal.raw);
    } else {
      newVal.l9[0] = (digitVal & 1) ? 1 : 0;
      newVal.l9[1] = (digitVal & 2) ? 1 : 0;
      newVal.l9[2] = (digitVal & 4) ? 1 : 0;
      newVal.l9[3] = (digitVal & 8) ? 1 : 0;
      return fromRaw<LogicCodeLength::Log9, 1>(newVal.raw);
    }
  }

  LOGIC_FULL *getValuePointer() {
    return isSelfContained() ? &VAL.raw : VAL.ptr;
  };
  const LOGIC_FULL *getValuePointer() const {
    return isSelfContained() ? &VAL.raw : VAL.ptr;
  };

  llvm::APInt toAPIntSlowCase() const;

  bool containsUnknownValuesSlowCase() const;
  static APLogicConstant concatSlowCase(const APLogicConstant &msb,
                                        const APLogicConstant &lsb);

  APLogicConstant lshiftSlowCase(unsigned shamt) const;

  bool isZeroLikeSlowCase() const;
};

llvm::hash_code hash_value(const APLogicConstant &Arg);

} // namespace hw
} // namespace circt

/// Provide DenseMapInfo for APLogicConstant.
namespace llvm {

template <>
struct DenseMapInfo<circt::hw::APLogicConstant, void> {
  static inline circt::hw::APLogicConstant getEmptyKey() {
    return circt::hw::APLogicConstant(
        circt::hw::LogicCodeLength::Log2,
        ~((circt::hw::APLogicConstant::LOGIC_FULL)0), 0);
  }

  static inline circt::hw::APLogicConstant getTombstoneKey() {
    return circt::hw::APLogicConstant(
        circt::hw::LogicCodeLength::Log2,
        ~((circt::hw::APLogicConstant::LOGIC_FULL)1), 0);
  }

  static unsigned getHashValue(const circt::hw::APLogicConstant &Key) {
    return static_cast<unsigned>(circt::hw::hash_value(Key));
  }

  static bool isEqual(const circt::hw::APLogicConstant &LHS,
                      const circt::hw::APLogicConstant &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

#endif // CIRCT_DIALECT_HW_APLOGICCONSTANT_H
