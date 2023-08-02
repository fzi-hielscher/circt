#ifndef CIRCT_DIALECT_HW_LOGIC_CODE_H
#define CIRCT_DIALECT_HW_LOGIC_CODE_H

// ********************************************************************
// *                    UNTESTED - DO NOT USE!                        *
// ********************************************************************

//                               |
//                _______________o_______________                  a[0]
//               |                               |
//        ______(0)______                 ______(1)______          a[1]
//       |               |               |               |
//    __(0)__         __(Z)__         __(1)__         __(X)__      a[2]
//   |       |       |       |       |       |       |       |
//  _o_     _o_     _o_     _o_     _o_     _o_     _o_     _o_    a[3]
// |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
// 0       L       Z       -       1       H       X   U   W

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdint.h>
#include <string.h>
#include <type_traits>

namespace circt {
namespace hw {
namespace logcode {

enum LogDigit : uint8_t {
  LOGD_0 = 0b0000,
  LOGD_L = 0b0100,
  LOGD_Z = 0b0010,
  LOGD_DC = 0b0110,
  LOGD_1 = 0b0001,
  LOGD_H = 0b0101,
  LOGD_X = 0b0011,
  LOGD_U = 0b1011,
  LOGD_W = 0b0111,
  LOGD_INVALID = 0b1111
};

static constexpr char logDigitToChar(LogDigit ldig) {
  switch (ldig) {
  case LogDigit::LOGD_0:
    return '0';
  case LogDigit::LOGD_L:
    return 'L';
  case LogDigit::LOGD_Z:
    return 'Z';
  case LogDigit::LOGD_DC:
    return '-';
  case LogDigit::LOGD_1:
    return '1';
  case LogDigit::LOGD_H:
    return 'H';
  case LogDigit::LOGD_X:
    return 'X';
  case LogDigit::LOGD_U:
    return 'U';
  case LogDigit::LOGD_W:
    return 'W';
  default:
    return '?';
  }
}

static constexpr LogDigit charToLogDigit(char c) {
  switch (c) {
  case '0':
    return LogDigit::LOGD_0;
  case 'L':
    return LogDigit::LOGD_L;
  case 'Z':
    return LogDigit::LOGD_Z;
  case '-':
    return LogDigit::LOGD_DC;
  case '1':
    return LogDigit::LOGD_1;
  case 'H':
    return LogDigit::LOGD_H;
  case 'X':
    return LogDigit::LOGD_X;
  case 'U':
    return LogDigit::LOGD_U;
  case 'W':
    return LogDigit::LOGD_W;
  default:
    return LogDigit::LOGD_INVALID;
  }
}

static constexpr bool isValidLogicDigit(LogDigit ldig) {
  switch (ldig) {
  case LogDigit::LOGD_0:
  case LogDigit::LOGD_L:
  case LogDigit::LOGD_Z:
  case LogDigit::LOGD_DC:
  case LogDigit::LOGD_1:
  case LogDigit::LOGD_H:
  case LogDigit::LOGD_X:
  case LogDigit::LOGD_U:
  case LogDigit::LOGD_W:
    return true;
  default:
    return false;
  }
}

static constexpr unsigned getCodeLengthForLogDigit(LogDigit digit) {
  switch (digit) {
  case LogDigit::LOGD_0:
  case LogDigit::LOGD_1:
    return 1;
  case LogDigit::LOGD_X:
  case LogDigit::LOGD_Z:
    return 2;
  default:
    return 4;
  }
}

template <LogDigit LD>
typename std::enable_if<isValidLogicDigit(LD), unsigned>::
    type static constexpr getCodeLengthForLogDigit() {
  return getCodeLengthForLogDigit(LD);
};

typedef uint8_t L9_8[4];
typedef uint16_t L9_16[4];
typedef uint32_t L9_32[4];
typedef uint64_t L9_64[4];

typedef uint8_t L4_8[2];
typedef uint16_t L4_16[2];
typedef uint32_t L4_32[2];
typedef uint64_t L4_64[2];

typedef uint8_t L2_8[1];
typedef uint16_t L2_16[1];
typedef uint32_t L2_32[1];
typedef uint64_t L2_64[1];

static constexpr bool _checkN(const unsigned N) {
  return N == 1 || N == 2 || N == 4;
}

static constexpr bool _checkBitwiseOperands(const unsigned NA,
                                            const unsigned NB) {
  return _checkN(NA) && _checkN(NB);
};

template <typename UIA, typename UIO = UIA>
typename std::enable_if<std::is_unsigned<UIO>::value &&
                            std::is_unsigned<UIA>::value,
                        UIO>::type static _getField_unsafe(const unsigned f,
                                                           const UIA *a,
                                                           const unsigned n) {
  return static_cast<UIO>((f < n) ? a[f] : 0);
}

template <unsigned F, typename UI, unsigned N>
typename std::enable_if<_checkN(N) && (F < 4 && F >= 0),
                        UI>::type static _getField(const UI (&a)[N]) {
  return _getField_unsafe(F, a, N);
}

template <unsigned F, typename UI, unsigned N>
typename std::enable_if<(N == 1 || N == 2 || N == 4) && (F < 4 && F >= 0),
                        void>::type static _clearFields(UI (&a)[N]) {
  for (unsigned i = F; i < N; i++)
    a[i] = (UI)0;
};

template <typename UI>
static size_t constexpr logCapacity(const UI (&a)[]) {
  return sizeof(UI) * 8;
}

template <typename UI>
static size_t constexpr logCapacity(const UI *a) {
  return sizeof(UI) * 8;
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4, void>::type static fromInt(
    UI (&a)[N], const UI val) {
  a[0] = val;
  _clearFields<1>(a);
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4, UI>::type static asInt(
    const UI (&a)[N]) {
  if (N > 1)
    assert(a[1] == 0);
  return a[0];
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4, UI>::type static toInt(
    const UI (&a)[N]) {
  return a[0];
}

template <typename UI>
static LogDigit digitAtIndex_dyn(const UI *a, const unsigned na,
                                 unsigned index) {
  assert(index < logCapacity(a));
  UI mask = ((UI)1) << index;

  uint8_t val = 0;
  val |= ((_getField_unsafe(0, a, na) & mask) != 0) ? 1 : 0;
  val |= ((_getField_unsafe(1, a, na) & mask) != 0) ? 2 : 0;
  val |= ((_getField_unsafe(2, a, na) & mask) != 0) ? 4 : 0;
  val |= ((_getField_unsafe(3, a, na) & mask) != 0) ? 8 : 0;

  return (LogDigit)val;
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4,
                        LogDigit>::type static digitAtIndex(const UI (&a)[N],
                                                            unsigned index) {
  return digitAtIndex_dyn(a, N, index);
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4,
                        bool>::type static fillWithValue(UI (&a)[N],
                                                         const LogDigit digit,
                                                         UI mask) {
  uint8_t val = (uint8_t)digit;
  for (unsigned i = 0; i < 4; i++) {
    if (i < N) {
      if ((val & (1 << i)) != 0)
        a[i] |= mask;
      else
        a[i] &= ~mask;
    } else {
      if ((val & (1 << i)) != 0)
        return false; // Digit not in logic
    }
  }
  return true;
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4, bool>::type static fromCStr(
    UI (&a)[N], const char s[], size_t length) {
  bool strValid = true;
  memset(a, 0, sizeof(UI) * N);
  for (size_t i = 0; i < length; i++) {
    LogDigit digit = charToLogDigit(s[i]);
    strValid &= digit != LogDigit::LOGD_INVALID;
    strValid &= fillWithValue(a, digit, (UI)(1ULL << (length - i - 1)));
  }
  return strValid;
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4, bool>::type static fromCStr(
    UI (&a)[N], const char s[]) {
  const size_t limit = logCapacity(a);
  size_t len = strnlen(s, limit);
  return fromCStr(a, s, len);
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4, unsigned>::
    type static countLeadingZeros(const UI (&a)[N]) {
  UI valueMask = a[0];
  if (N > 1)
    valueMask |= a[1];
  if (N > 2)
    valueMask |= a[2];

  UI shiftMask = ((UI)1) << (logCapacity(a) - 1);
  unsigned i;
  for (i = 0; i < logCapacity(a); i++) {
    if ((shiftMask & valueMask) != 0)
      break;
    shiftMask >>= 1;
  }

  return i;
}

template <typename UIA, typename UIB, unsigned NA, unsigned NB>
typename std::enable_if<_checkBitwiseOperands(NA, NB) && (NA >= NB) &&
                            (sizeof(UIA) >= sizeof(UIB)),
                        void>::type static copy(UIA (&dest)[NA],
                                                const UIB (&src)[NB]) {
  for (unsigned i = 0; i < NA; i++)
    dest[i] = _getField_unsafe<UIB, UIA>(i, src, NB);
}

template <typename UIA, typename UIB, unsigned NA, unsigned NB>
typename std::enable_if<_checkBitwiseOperands(NA, NB),
                        void>::type static copyAndCast(UIA (&dest)[NA],
                                                       const UIB (&src)[NB]) {
  for (unsigned i = 0; i < NA; i++)
    dest[i] = _getField_unsafe<UIB, UIA>(i, src, NB);
}

template <typename UIA, typename UIB, unsigned NA, unsigned NB>
typename std::enable_if<_checkBitwiseOperands(NA, NB),
                        void>::type static shiftCopyAndCast(UIA (&dest)[NA],
                                                            const UIB (
                                                                &src)[NB],
                                                            unsigned shamt) {
  for (unsigned i = 0; i < NA; i++)
    dest[i] = static_cast<UIA>((i < NB) ? (src[i] >> shamt) : 0);
}

template <typename UIA, typename UIB, unsigned NA, unsigned NB>
typename std::enable_if<_checkN(NA) && _checkN(NB) && NA >= NB,
                        void>::type static copySlice(UIA (&dest)[NA],
                                                     const UIB (&src)[NB],
                                                     unsigned srcOffset,
                                                     unsigned destOffset,
                                                     unsigned length = 1,
                                                     unsigned repl = 1) {
  if (length == 0 || repl == 0)
    return;

  assert((srcOffset + length) <= logCapacity(src));
  assert((destOffset + length * repl) <= logCapacity(dest));

  // Check if we can overwrite dest entirely
  if (length == logCapacity(dest)) {
    shiftCopyAndCast(dest, src, srcOffset);
    return;
  }

  UIA lengthMask = (((UIA)1) << length) - 1;

  // Create a mask with an one for each repetition at the correct bit offset
  UIA repetitionMask;
  if (length == 1) {
    // 'repl'-times ones
    repetitionMask = (~((UIA)0)) >> (logCapacity(dest) - repl);
  } else {
    // 'repl'-times ones with 'length - 1' zeros interleaved
    repetitionMask = 1;
    for (unsigned i = 1; i < repl; i++)
      repetitionMask |= repetitionMask << length;
  }

  UIA clearMask = (repetitionMask * lengthMask) << destOffset;
  // Patch bits into destination
  for (unsigned i = 0; i < NB; i++) {
    // Copy selected values into patchPre and right-align
    UIB patchPre = src[i] & ((UIB)lengthMask << srcOffset);
    patchPre >>= srcOffset;
    // Replicate values and shift into position
    UIA patch = (UIA)patchPre * repetitionMask;
    patch <<= destOffset;
    // Clear destination bits
    dest[i] &= ~clearMask;
    // Apply patch to destination
    dest[i] |= patch;
  }

  // Clear remaining bits if NA > NB
  for (unsigned i = NB; i < NA; i++) {
    dest[i] &= ~clearMask;
  }
}

template <typename UI, unsigned N>
typename std::enable_if<N == 2 || N == 4, UI>::type static getUnknownMask(
    const UI (&a)[N]) {
  return a[1];
}

template <typename UI, unsigned N>
typename std::enable_if<N == 4, UI>::type static getNon01XZMask(
    const UI (&a)[N]) {
  return a[2] | a[3];
}

template <typename UI, unsigned N>
typename std::enable_if<N == 2 || N == 4, UI>::type static getZMask(
    const UI (&a)[N]) {
  return ~a[0] & a[1] & ~_getField<2>(a);
}

template <typename UI>
static UI getLHMask(const UI (&a)[4]) {
  return ~a[1] & a[2];
}

// ------------------------
//  Conversions
// ------------------------

template <typename UI, unsigned NA, unsigned NB>
typename std::enable_if<(NA == 1 || NA == 2 || NA == 4) && (NB == 2 || NB == 4),
                        void>::type static convertTo01_XTo0(UI (&a)[NA],
                                                            const UI (&b)[NB]) {
  a[0] = b[0] & ~b[1];
  _clearFields<1>(a);
}

template <typename UI, unsigned NA, unsigned NB>
typename std::enable_if<(NA == 1 || NA == 2 || NA == 4) && (NB == 2 || NB == 4),
                        void>::type static convertTo01_XTo1(UI (&a)[NA],
                                                            const UI (&b)[NB]) {
  a[0] = b[0] | b[1];
  _clearFields<1>(a);
}

template <typename UI, unsigned NA, unsigned NB>
typename std::enable_if<(NA == 2 || NA == 4) && (NB == 2 || NB == 4),
                        void>::type static convertTo01X(UI (&a)[NA],
                                                        const UI (&b)[NB]) {
  a[0] = b[0] | b[1];
  a[1] = b[1];
  _clearFields<2>(a);
}

template <typename UI, unsigned N>
typename std::enable_if<N == 2 || N == 4, void>::type static convertTo01XZ(
    UI (&a)[N], const UI (&b)[4]) {
  a[0] = b[0] | (b[1] & b[2]);
  a[1] = b[1];
  _clearFields<2>(a);
}

template <typename UI>
static void convertTo01XU(UI (&a)[4], const UI (&b)[4]) {
  a[0] = b[0] | b[1];
  a[1] = b[1];
  a[2] = (UI)0;
  a[3] = b[3];
}

template <typename UI, unsigned NB>
typename std::enable_if<NB == 1 || NB == 2, void>::type static convertToL9(
    UI (&a)[4], const UI (&b)[NB]) {
  a[0] = b[0];
  a[1] = _getField<1>(b);
  a[2] = (UI)0;
  a[3] = (UI)0;
}

// ------------------------
//  Bitwise ops
// ------------------------

// XOR

template <typename UIAO, typename UIB>
static void opXor_unsafe(UIAO *o, const unsigned no, const UIAO *a,
                         const unsigned na, const UIB *b, const unsigned nb) {
  o[0] = a[0] ^ b[0];
  if (no > 1) {
    o[1] = _getField_unsafe(1, a, na) | _getField_unsafe<UIB, UIAO>(1, b, nb);
    o[0] |= o[1];
    if (no == 4) {
      o[2] = (UIAO)0;
      o[3] = _getField_unsafe(3, a, na) | _getField_unsafe<UIB, UIAO>(3, b, nb);
    }
  }
}

template <typename UIAO, typename UIB, unsigned NA, unsigned NB>
typename std::enable_if<_checkBitwiseOperands(NA, NB) && NA >= NB &&
                            sizeof(UIAO) >= sizeof(UIB),
                        void>::type static opXor_inplace(UIAO (&a)[NA],
                                                         const UIB (&b)[NB]) {
  opXor_unsafe(a, NA, a, NA, b, NB);
}

template <typename UI>
static void opXor_dyn(UI *o, const unsigned no, const UI *a, unsigned na,
                      const UI *b, unsigned nb) {
  assert(_checkBitwiseOperands(na, nb));
  assert(no == std::max(na, nb));
  opXor_unsafe(o, no, a, na, b, nb);
}

// OR

template <typename UI>
static void opOr_unsafe(UI *o, const unsigned no, const UI *a,
                        const unsigned na, const UI *b, const unsigned nb) {
  if (no > 1) {
    UI unknownBits = _getField_unsafe(1, a, na) & ~b[0];
    unknownBits |= ~a[0] & _getField_unsafe(1, b, nb);
    unknownBits |= _getField_unsafe(1, a, na) & _getField_unsafe(1, b, nb);
    o[1] = unknownBits;
  }
  o[0] = a[0] | b[0] | _getField_unsafe(1, o, no);
  if (no == 4) {
    o[2] = (UI)0;
    o[3] = o[1] & (_getField_unsafe(3, a, na) | _getField_unsafe(3, b, nb));
  }
}

template <typename UI, unsigned NA, unsigned NB>
typename std::enable_if<_checkBitwiseOperands(NA, NB) && NA >= NB,
                        void>::type static opOr_inplace(UI (&a)[NA],
                                                        const UI (&b)[NB]) {
  opOr_unsafe(a, NA, a, NA, b, NB);
}

template <typename UI>
static void opOr_dyn(UI *o, const unsigned no, const UI *a, unsigned na,
                     const UI *b, unsigned nb) {
  assert(_checkBitwiseOperands(na, nb));
  assert(no == std::max(na, nb));
  opOr_unsafe(o, no, a, na, b, nb);
}

// AND

template <typename UI>
static void opAnd_unsafe(UI *o, const unsigned no, const UI *a,
                         const unsigned na, const UI *b, const unsigned nb) {
  if (no > 1) {
    UI unknownBits = _getField_unsafe(1, a, na) & b[0];
    unknownBits |= a[0] & _getField_unsafe(1, b, nb);
    unknownBits |= _getField_unsafe(1, a, na) & _getField_unsafe(1, b, nb);
    o[1] = unknownBits;
  }
  o[0] = (a[0] & b[0]) | _getField_unsafe(1, o, no);
  if (no == 4) {
    o[2] = (UI)0;
    o[3] = o[1] & (_getField_unsafe(3, a, na) | _getField_unsafe(3, b, nb));
  }
}

template <typename UI, unsigned NA, unsigned NB>
typename std::enable_if<_checkBitwiseOperands(NA, NB) && NA >= NB,
                        void>::type static opAnd_inplace(UI (&a)[NA],
                                                         const UI (&b)[NB]) {
  opAnd_unsafe(a, NA, a, NA, b, NB);
}

template <typename UI>
static void opAnd_dyn(UI *o, const unsigned no, const UI *a, unsigned na,
                      const UI *b, unsigned nb) {
  assert(_checkBitwiseOperands(na, nb));
  assert(no == std::max(na, nb));
  opAnd_unsafe(o, no, a, na, b, nb);
}

// ------------------------
//  Shift Ops
// ------------------------

template <typename UI>
static void opShl_dyn(UI *a, const unsigned na, const unsigned shamt) {
  for (unsigned i = 0; i < na; i++)
    a[i] <<= shamt;
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4, void>::type static opShl(
    UI (&a)[N], const unsigned shamt) {
  opShl_dyn(a, N, shamt);
}

template <typename UI>
static void opShru_dyn(UI *a, const unsigned na, const unsigned shamt) {
  using UUI = std::make_unsigned_t<UI>;
  for (unsigned i = 0; i < na; i++)
    reinterpret_cast<UUI *>(a)[i] >>= shamt;
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4, void>::type static opShru(
    UI (&a)[N], const unsigned shamt) {
  opShru_dyn(a, N, shamt);
}

template <typename UI>
static void opShrs_dyn(UI *a, const unsigned na, const unsigned shamt) {
  using SUI = std::make_signed_t<UI>;
  for (unsigned i = 0; i < na; i++)
    reinterpret_cast<SUI *>(a)[i] >>= shamt;
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4, void>::type static opShrs(
    UI (&a)[N], const unsigned shamt) {
  opShrs_dyn(a, N, shamt);
}

// ------------------------
//  Reductions
// ------------------------

// XOR Reduce

template <typename UI>
static void reduceXor_unsafe(UI *o, const UI *a, const unsigned noa, UI mask,
                             unsigned destidx) {
  const UI outmask = ((UI)1) << destidx;
  if (_getField_unsafe(1, a, noa) & mask == 0) {
    // No unknown bits selected -> Result: 0/1
    auto par = __builtin_parityl(a[0] & mask);
    for (unsigned i = 0; i < noa; i++)
      o[i] &= ~outmask;
    if (par)
      o[0] |= outmask;
  } else {
    // Some selected bits unknown -> Result: X/U
    o[0] |= outmask;
    o[1] |= outmask;
    o[2] &= ~outmask;
    if (noa == 4) {
      if ((a[3] & mask) != 0)
        o[3] |= outmask;
      else
        o[3] &= ~outmask;
    }
  }
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4,
                        void>::type static reduceXor(UI (&o)[N],
                                                     const UI (&a)[N], UI mask,
                                                     unsigned destidx = 0) {
  reduceXor_unsafe(o, a, N, mask, destidx);
}

template <typename UI>
static void reduceXor_dyn(UI *o, const UI *a, const unsigned noa, UI mask,
                          unsigned destidx = 0) {
  assert(noa == 1 || noa == 2 || noa == 4);
  reduceXor_unsafe(o, a, noa, mask, destidx);
}

// OR Reduce

template <typename UI>
static void reduceOr_unsafe(UI *o, const UI *a, const unsigned noa, UI mask,
                            unsigned destidx) {
  const UI outmask = ((UI)1) << destidx;
  UI nonZeroMask = a[0];
  nonZeroMask &= ~_getField_unsafe(1, a, noa);
  if ((nonZeroMask & mask) == 0) {
    // All selected bits zero or unknown
    if ((_getField_unsafe(1, a, noa) & mask) == 0) {
      // All selected bits zero -> Result : 0
      for (unsigned i = 0; i < noa; i++)
        o[0] &= ~outmask;
    } else {
      // Some selected bits unknown -> Result : X/U
      o[0] |= outmask;
      o[1] |= outmask;
      if (noa == 4) {
        o[2] &= ~outmask;
        o[3] &= ~outmask;
        if ((a[3] & mask) != 0)
          o[3] |= outmask;
      }
    }
  } else {
    // Some selected bits one -> Result : 1
    o[0] |= outmask;
    for (unsigned i = 1; i < noa; i++)
      o[i] &= ~outmask;
  }
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4, void>::type static reduceOr(
    UI (&o)[N], const UI (&a)[N], UI mask, unsigned destidx = 0) {
  reduceOr_unsafe(o, a, N, mask, destidx);
}

template <typename UI>
static void reduceOr_dyn(UI *o, const UI *a, const unsigned noa, UI mask,
                         unsigned destidx = 0) {
  assert(noa == 1 || noa == 2 || noa == 4);
  reduceOr_unsafe(o, a, noa, mask, destidx);
}

// AND Reduce

template <typename UI>
static void reduceAnd_unsafe(UI *o, const UI *a, const unsigned noa, UI mask,
                             unsigned destidx) {
  const UI outmask = ((UI)1) << destidx;
  UI onesMask = a[0];
  onesMask |= _getField_unsafe(1, a, noa);
  if ((onesMask & mask) == mask) {
    // All selected bits one or unknown
    if ((_getField_unsafe(1, a, noa) & mask) == 0) {
      // All selected bits one -> Result : 1
      o[0] |= outmask;
      for (unsigned i = 1; i < noa; i++)
        o[i] &= ~outmask;
    } else {
      // Some selected bits unknown -> Result : X/U
      o[0] |= outmask;
      o[1] |= outmask;
      if (noa == 4) {
        o[2] &= ~outmask;
        o[3] &= ~outmask;
        if ((a[3] & mask) != 0)
          o[3] |= outmask;
      }
    }
  } else {
    // Some selected bits zero -> Result : 0
    for (unsigned i = 0; i < noa; i++)
      o[0] &= ~outmask;
  }
}

template <typename UI, unsigned N>
typename std::enable_if<N == 1 || N == 2 || N == 4,
                        void>::type static reduceAnd(UI (&o)[N],
                                                     const UI (&a)[N], UI mask,
                                                     unsigned destidx = 0) {
  reduceAnd_unsafe(o, a, N, mask, destidx);
}

template <typename UI>
static void reduceAnd_dyn(UI *o, const UI *a, const unsigned noa, UI mask,
                          unsigned destidx = 0) {
  assert(noa == 1 || noa == 2 || noa == 4);
  reduceAnd_unsafe(o, a, noa, mask, destidx);
}

} // namespace logcode
} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_LOGIC_CODE_H
