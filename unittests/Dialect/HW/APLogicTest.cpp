#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWAPLogic.h"
#include "circt/Dialect/HW/HWAPLogicConstant.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

static const char L9CHARS[10] = {'U', 'X', '0', '1', 'Z', 'W', 'L', 'H', '-', '\0'};
static const char L4CHARS[5] =  {'X', '0', '1', 'Z', '\0'};
static const char L2CHARS[3] =  {'0', '1', '\0'};

static const char RANDOML9[401] = "1-XU1WZZWULWHU0Z00WLZXU0-ZLH1LH--1-UUZ01WX1WWLZWZW0LH-X1U111-WULZ-WZWHH1111H1U1L-1W-ZUHL0XZL-0W1X0UZXLL-00ZUUXLZ10L0L0ZUH1HXXZLLWL0-010-X0ZWU0X1ZX-XWL1XLH--0WL001ZLLLUXUZ-XZ-ZL-UZW1HW-U-U-W-HUUZWZ1U10HXZ0ZH1HLUH-1ZLLULU0HWUUHHHH0HZLU1L0110-UWZ1L01XHXHH---X0HLXZ-LHHLL-WLU-W-H-WZUXHZX-0XWLU1UU0UX1H-XW1XH-1WHLZXZWH1-ULXWZ1ZUL01WX0W1L-XHHZ10W-1HWHLXLX0HZWWULWWZZ-00-LXUUW0XX1-Z1-UW-ZLZZLLU1UULLWLWX0H1U";
static const char RANDOML4[401] = "00XZ11Z1X010ZZ10XZX0X11Z0110X0Z1Z1ZZ010ZXX1XXZ001ZXX10ZXZ00X1XXXZXZXZX11Z11Z110XZ0ZX00XZZ0101XXZ01XZ110ZZ11ZX1XZZZXXZXZ1XX110XX0XZZX0011ZZZ1101X1ZZXZ0Z1Z0X11101Z1XXZ11ZXX0ZZ0Z1XXZ1X001ZZ0Z1ZX1XXZZ1011XZZXZZXX1010XX0ZX00ZZ000ZXX00Z0XZZZX0ZXXZ1ZXZ1X0XZXX01X11XX01ZZZXZZ01ZZ000XX1X00X0Z11ZXXXXXX10X00100ZXZZXX11X1Z0ZX01011XXX11ZZZXX11X1X10XZZZ100X0Z1XX0ZZZZXXZ0XZZXZ1XXZ01Z100Z001X0X1ZZ00X10Z01ZZZ1ZXX1X";
static const char RANDOML2[401] = "0000010101100110100011111111100011000111001100111101011101000001101111100010111011010100000101000111101110001101110011110001010111101000110000010010111001000111000010000100001100010010000110101000110110111100111001110000110110001111011001001110011101101110110011111101111100010000001011101001100001100110100110101100000100100110001000001101010001001000111111100000110001011111010100111011011001011001";

static constexpr uint8_t charToIdx(const char c) {
  switch (c) {
    case 'U':
      return 0;
    case 'X':
      return 1;
    case '0':
      return 2;
    case '1':
      return 3;
    case 'Z':
      return 4;
    case 'W':
      return 5;
    case 'L':
      return 6;
    case 'H':
      return 7;
    case '-':
      return 8;
    default:
      return 255;
  }
}

static const char LUT_XOR[9][9] = {
  {'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U'},
  {'U', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'},
  {'U', 'X', '0', '1', 'X', 'X', '0', '1', 'X'},
  {'U', 'X', '1', '0', 'X', 'X', '1', '0', 'X'},
  {'U', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'},
  {'U', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'},
  {'U', 'X', '0', '1', 'X', 'X', '0', '1', 'X'},
  {'U', 'X', '1', '0', 'X', 'X', '1', '0', 'X'},
  {'U', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'}
};

static std::string xorRef(const std::string &a, const std::string &b) {
  std::string result(a.size(), '\0');
  for (size_t i = 0; i < a.size(); i++) {
    result[i] = LUT_XOR[charToIdx(a[i])][charToIdx(b[i])];
  }
  return result;
}

static const unsigned TEST_LENGTHS[] =
 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  17, 18, 19, 23, 30, 31, 32, 33, 34, 40, 42, 51, 62,
  63, 64, 65, 66, 72, 100, 111, 127, 128,
  129, 150, 191, 192, 193, 200, 250,
  255, 256, 257, 300, 320, 321};

static LogicCodeLength minimumCode(const std::string& str) {
  auto l9ExSet = std::set<char>({'U', 'W', 'L', 'H', '-'});
  bool isL2 = true;
  bool isL4 = true;
  for(char c: str) {
    if (c == '0' || c == '1')
      continue;
    if (c == 'X' || c == 'Z') {
      isL2 = false;
      continue;
    }
    isL2 = false;
    isL4 = false;
    EXPECT_TRUE(l9ExSet.count(c) > 0);
    break;
  }
  if (isL2)
    return LogicCodeLength::Log2;
  else if (isL4)
    return LogicCodeLength::Log4;
  else
    return LogicCodeLength::Log9;
}


TEST(APLogicTest, APLogicZeroTest) {
  for (unsigned n: TEST_LENGTHS) {
    APLogic zero(n);
    EXPECT_EQ(zero.getBitWidth(), n);
    size_t nSegements = n / 64;
    if (n % 64 != 0)
      nSegements++;
    EXPECT_STREQ(zero.toString().c_str(), std::string(n, '0').c_str());
  }
}

TEST(APLogicTest, NoZeroLengthTest) {
  EXPECT_DEBUG_DEATH(APLogic(0), "zero length logic values not allowed");
  EXPECT_DEBUG_DEATH(APLogic("", 0), "zero length logic values not allowed");
  EXPECT_DEBUG_DEATH(APLogic(0, 0), "zero length logic values not allowed");
  EXPECT_DEBUG_DEATH(APLogic(0, 12345), "zero length logic values not allowed");
}

static void apLogicStringTest(const char * cptr, unsigned len, unsigned offset, unsigned minLength) {
  std::string testStr(cptr + offset, len); 
  APLogic apl(testStr.c_str(), minLength);
  auto actLength = (len < minLength) ? minLength : len; 
  EXPECT_EQ(apl.getBitWidth(), actLength);
  size_t nSegements = actLength / 64;
  if (actLength % 64 != 0)
    nSegements++;
  if (actLength != len)
    testStr.insert(0, std::string(actLength - len, '0'));
  EXPECT_STREQ(apl.toString().c_str(), testStr.c_str());
}

TEST(APLogicTest, APLogicParsePrintTest) {

  APLogic l9vals = APLogic(std::string(L9CHARS));
  EXPECT_EQ(l9vals.getBitWidth(), 9U);
  EXPECT_STREQ(l9vals.toString().c_str(), L9CHARS);

  APLogic l9valsNoExt = APLogic(std::string(L9CHARS), 8);
  EXPECT_EQ(l9valsNoExt.getBitWidth(), 9U);
  EXPECT_STREQ(l9valsNoExt.toString().c_str(), L9CHARS);

  APLogic zero1 = APLogic("", 1);
  EXPECT_EQ(zero1.getBitWidth(), 1U);
  EXPECT_STREQ(zero1.toString().c_str(), "0");


  apLogicStringTest(RANDOML9, 3, 0, 63);
  apLogicStringTest(RANDOML9, 3, 1, 64);
  apLogicStringTest(RANDOML9, 3, 2, 65);
  apLogicStringTest(RANDOML9, 15, 3, 16);
  apLogicStringTest(RANDOML9, 16, 4, 17);
  apLogicStringTest(RANDOML9, 126, 5, 500);

  for (unsigned n: TEST_LENGTHS) {
    apLogicStringTest(RANDOML2, n, n % 13, 0);
    apLogicStringTest(RANDOML4, n, n % 13, 0);
    apLogicStringTest(RANDOML9, n, n % 13, 0);
  }
  
}

TEST(APLogicTest, DigitSetGetTest) {
  APLogic aplSmall(62);
  for (unsigned i = 0; i < 62; i++) {
    auto dig = logcode::charToLogDigit(RANDOML9[i]);
    aplSmall.setDigitAtIndex(dig, 62 - i - 1);    
  }
  EXPECT_STREQ(aplSmall.toString().c_str(), std::string(RANDOML9, 62).c_str());
  for (unsigned i = 0; i < 62; i++) {
    auto ref = logcode::charToLogDigit(RANDOML9[i]);
    EXPECT_EQ(aplSmall.getDigitAtIndex(62 - i - 1), ref);   
  }

  APLogic aplLarge(192);
  for (unsigned i = 0; i < 192; i++) {
    auto dig = logcode::charToLogDigit(RANDOML9[i]);
    aplLarge.setDigitAtIndex(dig, 192 - i - 1);    
  }
  EXPECT_STREQ(aplLarge.toString().c_str(), std::string(RANDOML9, 192).c_str());
  for (unsigned i = 0; i < 192; i++) {
    auto ref = logcode::charToLogDigit(RANDOML9[i]);
    EXPECT_EQ(aplLarge.getDigitAtIndex(192 - i - 1), ref);   
  }
}

TEST(APLogicTest, APMinimumCodeTest) {
  EXPECT_EQ(APLogic("000").getMinimumRequiredCode(), LogicCodeLength::Log2);
  EXPECT_EQ(APLogic("010").getMinimumRequiredCode(), LogicCodeLength::Log2);
  EXPECT_EQ(APLogic("0X0").getMinimumRequiredCode(), LogicCodeLength::Log4);
  EXPECT_EQ(APLogic("0Z0").getMinimumRequiredCode(), LogicCodeLength::Log4);
  EXPECT_EQ(APLogic("0W0").getMinimumRequiredCode(), LogicCodeLength::Log9);
  EXPECT_EQ(APLogic("0-0").getMinimumRequiredCode(), LogicCodeLength::Log9);
  EXPECT_EQ(APLogic("0U0").getMinimumRequiredCode(), LogicCodeLength::Log9);
  EXPECT_EQ(APLogic("0L0").getMinimumRequiredCode(), LogicCodeLength::Log9);
  EXPECT_EQ(APLogic("0H0").getMinimumRequiredCode(), LogicCodeLength::Log9);

  std::string testStr(250, '0');
  EXPECT_EQ(APLogic(testStr).getMinimumRequiredCode(), LogicCodeLength::Log2);
  testStr[63] = 'X';
  EXPECT_EQ(APLogic(testStr).getMinimumRequiredCode(), LogicCodeLength::Log4);
  testStr[199] = '-';
  EXPECT_EQ(APLogic(testStr).getMinimumRequiredCode(), LogicCodeLength::Log9);
  testStr[63] = '1';
  EXPECT_EQ(APLogic(testStr).getMinimumRequiredCode(), LogicCodeLength::Log9);
  testStr[199] = '1';
  EXPECT_EQ(APLogic(testStr).getMinimumRequiredCode(), LogicCodeLength::Log2);
  testStr[249] = 'Z';
  EXPECT_EQ(APLogic(testStr).getMinimumRequiredCode(), LogicCodeLength::Log4);
  testStr[0] = 'L';
  EXPECT_EQ(APLogic(testStr).getMinimumRequiredCode(), LogicCodeLength::Log9);
}

TEST(APLogicTest, CopyAndMoveTest) {
  auto smallRef = std::string(RANDOML9, 63);
  auto largeRef = std::string(RANDOML9 + 7, 300);
  APLogic smallOrig(smallRef);
  APLogic largeOrig(largeRef);
  APLogic smallCopy(smallOrig);
  APLogic largeCopy(largeOrig);
  EXPECT_EQ(smallOrig.getBitWidth(), 63U);
  EXPECT_EQ(largeOrig.getBitWidth(), 300U);
  EXPECT_EQ(smallOrig.getBitWidth(), smallCopy.getBitWidth());
  EXPECT_EQ(largeOrig.getBitWidth(), largeCopy.getBitWidth());
  EXPECT_STREQ(smallCopy.toString().c_str(), smallRef.c_str());
  EXPECT_STREQ(largeCopy.toString().c_str(), largeRef.c_str());

  
  APLogic smallCopyAssigned(1);
  smallCopyAssigned = smallOrig;
  EXPECT_EQ(smallCopyAssigned.getBitWidth(), 63U);
  EXPECT_STREQ(smallCopyAssigned.toString().c_str(), smallRef.c_str());


  APLogic smallAnother(500);
  APLogic largeAnother(500);
  APLogic notQuiteAsLarge(largeRef.substr(30, 200));

  smallAnother = smallOrig;
  largeAnother = largeOrig;
  notQuiteAsLarge = largeAnother;
  EXPECT_EQ(smallOrig.getBitWidth(), 63U);
  EXPECT_EQ(largeOrig.getBitWidth(), 300U);
  EXPECT_EQ(smallOrig.getBitWidth(), smallAnother.getBitWidth());
  EXPECT_EQ(largeOrig.getBitWidth(), largeAnother.getBitWidth());
  EXPECT_EQ(largeAnother.getBitWidth(), notQuiteAsLarge.getBitWidth());
  EXPECT_STREQ(smallAnother.toString().c_str(), smallRef.c_str());
  EXPECT_STREQ(largeAnother.toString().c_str(), largeRef.c_str());
  EXPECT_STREQ(notQuiteAsLarge.toString().c_str(), largeRef.c_str());
  
  APLogic smallMoved(std::move(smallCopy));
  APLogic largeMoved(std::move(largeCopy));
  EXPECT_EQ(smallCopy.getBitWidth(), 0U);
  EXPECT_EQ(largeCopy.getBitWidth(), 0U);
  EXPECT_STREQ(smallMoved.toString().c_str(), smallRef.c_str());
  EXPECT_STREQ(largeMoved.toString().c_str(), largeRef.c_str());

  smallMoved = APLogic(3);
  largeMoved = APLogic(132);
  EXPECT_EQ(smallMoved.getBitWidth(), 3U);
  EXPECT_EQ(largeMoved.getBitWidth(), 132U);
  EXPECT_STREQ(smallMoved.toString().c_str(), std::string(3, '0').c_str());
  EXPECT_STREQ(largeMoved.toString().c_str(), std::string(132, '0').c_str());

  smallMoved = std::move(smallAnother);
  largeMoved = std::move(largeAnother);
  EXPECT_EQ(smallOrig.getBitWidth(), smallMoved.getBitWidth());
  EXPECT_EQ(largeOrig.getBitWidth(), largeMoved.getBitWidth());
  EXPECT_EQ(smallAnother.getBitWidth(), 0U);
  EXPECT_EQ(largeAnother.getBitWidth(), 0U);
  EXPECT_STREQ(smallMoved.toString().c_str(), smallRef.c_str());
  EXPECT_STREQ(largeMoved.toString().c_str(), largeRef.c_str());
}

static void replTest(APLogic &dest, const APLogic &src, unsigned index, unsigned length, unsigned repeat, unsigned srcOffset) {
  auto refString = dest.toString();
  if (repeat > 0 && length > 0) {
    auto srcString = src.toString();
    auto subString = srcString.substr(
      srcString.size() - srcOffset - length, length);
    for (unsigned int i = 0; i < repeat; i++) {
      refString.replace(
        refString.size() - index - ((i + 1) * length),
        length, subString);
    }
  }
  dest.replace(src, index, length, repeat, srcOffset);
  EXPECT_STREQ(dest.toString().c_str(), refString.c_str());
}

static void replTest(APLogic &dest, const APLogicConstant &src, unsigned index, unsigned length, unsigned repeat, unsigned srcOffset) {
  auto refString = dest.toString();
  if (repeat > 0 && length > 0) {
    auto srcString = APLogic(src).toString();
    auto subString = srcString.substr(
      srcString.size() - srcOffset - length, length);
    for (unsigned int i = 0; i < repeat; i++) {
      refString.replace(
        refString.size() - index - ((i + 1) * length),
        length, subString);
    }
  }
  dest.replace(src, index, length, repeat, srcOffset);
  EXPECT_STREQ(dest.toString().c_str(), refString.c_str());
}


TEST(APLogicTest, APLogicReplaceTest) {
  auto smallDest = APLogic(64);
  auto smallSrc = APLogic("1X0ZWLU1");
  replTest(smallDest, smallSrc, 0, smallSrc.getBitWidth(), 1, 0);
  replTest(smallDest, smallSrc, 3, 4, 1, 2);
  replTest(smallDest, smallSrc, 4, 3, 15, 1);
  replTest(smallDest, smallSrc, 0, 8, 8, 0);
  replTest(smallDest, smallSrc, 0, 0, 8, 0);
  replTest(smallDest, smallSrc, 0, 8, 0, 0);

  auto largeApl = APLogic(264);
  replTest(largeApl, smallSrc, 0, 8, 264 / 8, 0);
  replTest(largeApl, smallSrc, 63, 7, 5, 1);
  replTest(largeApl, smallSrc, 261, 3, 1, 3);

  auto anotherlargeApl = APLogic(180);

  replTest(anotherlargeApl, APLogic("-0XZUWZL1XX1H1U-"), 0, 15, 180 / 15, 1);

  replTest(anotherlargeApl, anotherlargeApl, 1, 71, 1, 0);
  replTest(anotherlargeApl, anotherlargeApl, 71, 71, 1, 0);
  replTest(anotherlargeApl, anotherlargeApl, 71, 71, 1, 1);

  replTest(anotherlargeApl, anotherlargeApl, 0, 4, 21, 84);
  replTest(anotherlargeApl, anotherlargeApl, 0, 5, 21, 84);
  
  replTest(largeApl, largeApl, 64, 64, 3, 0);
  replTest(largeApl, anotherlargeApl, 20, 180, 1, 0);

  replTest(largeApl, largeApl, 0, 200, 1, 64);  
}

TEST(APLogicConstantTest, LudicrouslyLongLogicTest) {
  auto longLogic = APLogic(1UL << 29);
  EXPECT_EQ(longLogic.getBitWidth(), 1UL << 29);

  auto longConst2 = APLogicConstant(longLogic);
  EXPECT_EQ(longConst2.getBitWidth(), 1UL << 29);
  EXPECT_EQ(longConst2.getCodeLength(), LogicCodeLength::Log2);

  longLogic.setDigitAtIndex(logcode::LogDigit::LOGD_Z, (1UL << 27) + 35);
  auto longConst4 = APLogicConstant(longLogic);
  EXPECT_EQ(longConst4.getBitWidth(), 1UL << 29);
  EXPECT_EQ(longConst4.getCodeLength(), LogicCodeLength::Log4);

  longLogic.setDigitAtIndex(logcode::LogDigit::LOGD_W, (1UL << 13) - 3);
  auto longConst9 = APLogicConstant(longLogic);
  EXPECT_EQ(longConst9.getBitWidth(), 1UL << 29);
  EXPECT_EQ(longConst9.getCodeLength(), LogicCodeLength::Log9);
}

TEST(APLogicConstantTest, CharTemplateConstructorTest) {
  auto dig0 = APLogicConstant::fromChar<'0'>();
  EXPECT_EQ(dig0.getBitWidth(), 1U);
  EXPECT_EQ(dig0.getCodeLength(), LogicCodeLength::Log2);
  EXPECT_STREQ(APLogic(dig0).toString().c_str(), "0");

  auto dig1 = APLogicConstant::fromChar<'1'>();
  EXPECT_EQ(dig1.getBitWidth(), 1U);
  EXPECT_EQ(dig1.getCodeLength(), LogicCodeLength::Log2);
  EXPECT_STREQ(APLogic(dig1).toString().c_str(), "1");

  auto digX = APLogicConstant::fromChar<'X'>();
  EXPECT_EQ(digX.getBitWidth(), 1U);
  EXPECT_EQ(digX.getCodeLength(), LogicCodeLength::Log4);
  EXPECT_STREQ(APLogic(digX).toString().c_str(), "X");

  auto digZ = APLogicConstant::fromChar<'Z'>();
  EXPECT_EQ(digZ.getBitWidth(), 1U);
  EXPECT_EQ(digZ.getCodeLength(), LogicCodeLength::Log4);
  EXPECT_STREQ(APLogic(digZ).toString().c_str(), "Z");

  auto digU = APLogicConstant::fromChar<'U'>();
  EXPECT_EQ(digU.getBitWidth(), 1U);
  EXPECT_EQ(digU.getCodeLength(), LogicCodeLength::Log9);
  EXPECT_STREQ(APLogic(digU).toString().c_str(), "U");

  auto digW = APLogicConstant::fromChar<'W'>();
  EXPECT_EQ(digW.getBitWidth(), 1U);
  EXPECT_EQ(digW.getCodeLength(), LogicCodeLength::Log9);
  EXPECT_STREQ(APLogic(digW).toString().c_str(), "W");

  auto digH = APLogicConstant::fromChar<'H'>();
  EXPECT_EQ(digH.getBitWidth(), 1U);
  EXPECT_EQ(digH.getCodeLength(), LogicCodeLength::Log9);
  EXPECT_STREQ(APLogic(digH).toString().c_str(), "H");

  auto digL = APLogicConstant::fromChar<'L'>();
  EXPECT_EQ(digL.getBitWidth(), 1U);
  EXPECT_EQ(digL.getCodeLength(), LogicCodeLength::Log9);
  EXPECT_STREQ(APLogic(digL).toString().c_str(), "L");

  auto digDC = APLogicConstant::fromChar<'-'>();
  EXPECT_EQ(digDC.getBitWidth(), 1U);
  EXPECT_EQ(digDC.getCodeLength(), LogicCodeLength::Log9);
  EXPECT_STREQ(APLogic(digDC).toString().c_str(), "-");
}

template <char C>
static void getFilledTest(unsigned n, LogicCodeLength logCode) {
  auto aplc = APLogicConstant::getFilled<C>(n);
  EXPECT_EQ(aplc.getBitWidth(), n);
  EXPECT_EQ(aplc.getCodeLength(), logCode);
  EXPECT_STREQ(APLogic(aplc).toString().c_str(), std::string(n, C).c_str());
}

TEST(APLogicConstantTest, GetFilledShortConstructorTest) {
  for (auto n: TEST_LENGTHS) {
    getFilledTest<'0'>(n, LogicCodeLength::Log2);
    getFilledTest<'1'>(n, LogicCodeLength::Log2);
    getFilledTest<'X'>(n, LogicCodeLength::Log4);
    getFilledTest<'Z'>(n, LogicCodeLength::Log4);
    getFilledTest<'W'>(n, LogicCodeLength::Log9);
    getFilledTest<'U'>(n, LogicCodeLength::Log9);
    getFilledTest<'-'>(n, LogicCodeLength::Log9);
    getFilledTest<'L'>(n, LogicCodeLength::Log9);
    getFilledTest<'H'>(n, LogicCodeLength::Log9);
  }
}

TEST(APLogicTest, APLogicXorTest) {
  auto apl9A = APLogic(std::string(L9CHARS, 9), 81);
  apl9A.replace(apl9A, 9, 9, 8, 0);
  auto apl9B = APLogic(81);
  for (unsigned i = 0; i < 9; i++)
    apl9B.replace(
      APLogicConstant::getFilled(logcode::charToLogDigit(L9CHARS[i]), 9),
      i * 9);
  apl9A ^= apl9B;
  auto strRes = apl9A.toString();
  for (unsigned i = 0; i < 9; i++)
    EXPECT_STREQ(strRes.substr(i * 9, 9).c_str(), std::string(LUT_XOR[9 - i - 1], 9).c_str());


  for (unsigned i = 0; i < 9; i++)  {
    auto apl = APLogic(APLogicConstant(std::string(L9CHARS, 9)));
    auto apc = APLogicConstant::getFilled(logcode::charToLogDigit(L9CHARS[i]), 9);     
    apl ^= apc;
    EXPECT_STREQ(apl.toString().c_str(), std::string(LUT_XOR[i], 9).c_str());
  }


  auto apcRand = APLogicConstant(std::string(RANDOML2 + 21, 212));
  auto aplRand = APLogic(apcRand);
  aplRand ^= apcRand;
  EXPECT_TRUE(APLogicConstant(aplRand).isZero());

  for (unsigned n: TEST_LENGTHS) {
    auto apl = APLogic(std::string(RANDOML9 + (n % 13), n));
    auto apc9 = APLogicConstant(APLogic(std::string(RANDOML9 + (n % 7), n)));
    auto apc4 = APLogicConstant(APLogic(std::string(RANDOML4 + (n % 7), n)));
    auto apc2 = APLogicConstant(APLogic(std::string(RANDOML2 + (n % 7), n)));

    APLogic apl9(apl);
    apl9 ^= apc9;
    EXPECT_STREQ(apl9.toString().c_str(), xorRef(apl.toString(), APLogic(apc9).toString()).c_str());

    APLogic apl4(apl);
    apl4 ^= apc4;
    EXPECT_STREQ(apl4.toString().c_str(), xorRef(apl.toString(), APLogic(apc4).toString()).c_str());

    APLogic apl2(apl);
    apl2 ^= apc2;
    EXPECT_STREQ(apl2.toString().c_str(), xorRef(apl.toString(), APLogic(apc2).toString()).c_str());
  }
}

TEST(APLogicConstantTest, APLogicConversionTest) {

  // L2
  for (auto n: TEST_LENGTHS) {
    auto l2ref = std::string(RANDOML2 + (n % 7), n);
    auto l2Orig = APLogic(l2ref);
    auto l2Const = APLogicConstant(l2Orig);
    EXPECT_EQ(l2Const.getBitWidth(), n);
    EXPECT_EQ(l2Const.getCodeLength(), LogicCodeLength::Log2);
    auto l2New = APLogic(l2Const);
    EXPECT_EQ(l2New.getBitWidth(), n);
    EXPECT_STREQ(l2New.toString().c_str(), l2ref.c_str());
  }

  // L4
  for (auto n: TEST_LENGTHS) {
    auto l4ref = std::string(RANDOML4 + (n % 7), n);
    auto l4Orig = APLogic(l4ref);
    auto l4Const = APLogicConstant(l4Orig);
    EXPECT_EQ(l4Const.getBitWidth(), n);
    EXPECT_EQ(l4Const.getCodeLength(), minimumCode(l4ref));
    auto l4New = APLogic(l4Const);
    EXPECT_EQ(l4New.getBitWidth(), n);
    EXPECT_STREQ(l4New.toString().c_str(), l4ref.c_str());
  }

  // L9
  for (auto n: TEST_LENGTHS) {
    auto l9ref = std::string(RANDOML9 + (n % 7), n);
    auto l9Orig = APLogic(l9ref);
    auto l9Const = APLogicConstant(l9Orig);
    EXPECT_EQ(l9Const.getBitWidth(), n);
    EXPECT_EQ(l9Const.getCodeLength(), minimumCode(l9ref));
    auto l9New = APLogic(l9Const);
    EXPECT_EQ(l9New.getBitWidth(), n);
    EXPECT_STREQ(l9New.toString().c_str(), l9ref.c_str());
  }
}

TEST(APLogicTest, APLogicConstantExpansionTest) {

  auto strRef = std::string(256, '0');

  APLogic apl1 = APLogic(APLogicConstant::fromChar<'1'>(), 256);
  strRef[255] = '1';
  EXPECT_EQ(apl1.getBitWidth(), 256U);
  EXPECT_STREQ(apl1.toString().c_str(), strRef.c_str());

  APLogic aplZ = APLogic(APLogicConstant::fromChar<'Z'>(), 256);
  strRef[255] = 'Z';
  EXPECT_EQ(aplZ.getBitWidth(), 256U);
  EXPECT_STREQ(aplZ.toString().c_str(), strRef.c_str());

  APLogic aplU = APLogic(APLogicConstant::fromChar<'U'>(), 256);
  strRef[255] = 'U';
  EXPECT_EQ(aplU.getBitWidth(), 256U);
  EXPECT_STREQ(aplU.toString().c_str(), strRef.c_str());

  strRef = std::string(67, 'X');
  strRef[55] = 'Z'; 
  auto apc = APLogicConstant(APLogic(strRef));
  APLogic apl0X = APLogic(apc, 67U + 333U);
  strRef.insert(0, std::string(333, '0'));
  EXPECT_EQ(apl0X.getBitWidth(), 67U + 333U);
  EXPECT_STREQ(apl0X.toString().c_str(), strRef.c_str());

  // L2
  for (auto n: TEST_LENGTHS) {
    auto l2ref = std::string(RANDOML2 + (n % 7), n);
    auto l2Orig = APLogic(l2ref);
    auto l2Const = APLogicConstant(l2Orig);
    auto l2New = APLogic(l2Const, n + 1);
    EXPECT_EQ(l2New.getBitWidth(), n + 1);
    l2ref.insert(0, "0");
    EXPECT_STREQ(l2New.toString().c_str(), l2ref.c_str());
  }

  // L4
  for (auto n: TEST_LENGTHS) {
    auto l4ref = std::string(RANDOML4 + (n % 7), n);
    auto l4Orig = APLogic(l4ref);
    auto l4Const = APLogicConstant(l4Orig);
    auto l4New = APLogic(l4Const, n + 1);
    EXPECT_EQ(l4New.getBitWidth(), n + 1);
    l4ref.insert(0, "0");
    EXPECT_STREQ(l4New.toString().c_str(), l4ref.c_str());
  }

  // L9
  for (auto n: TEST_LENGTHS) {
    auto l9ref = std::string(RANDOML9 + (n % 7), n);
    auto l9Orig = APLogic(l9ref);
    auto l9Const = APLogicConstant(l9Orig);
    auto l9New = APLogic(l9Const, n + 1);
    EXPECT_EQ(l9New.getBitWidth(), n + 1);
    l9ref.insert(0, "0");
    EXPECT_STREQ(l9New.toString().c_str(), l9ref.c_str());
  }
}

TEST(APLogicConstantTest, APIntConversionTest) {
  for (auto n: TEST_LENGTHS) {
    auto origInt = APInt(n, llvm::StringRef(RANDOML2 + (n % 5), n), 2);
    auto logConst = APLogicConstant(origInt);
    EXPECT_EQ(logConst.getBitWidth(), n);
    EXPECT_EQ(logConst.getCodeLength(), LogicCodeLength::Log2);
    EXPECT_TRUE(logConst.isInteger());
    EXPECT_TRUE(logConst.isIntegerLike());
    auto asInt = logConst.asAPInt();
    auto toInt = logConst.toAPInt();
    EXPECT_EQ(asInt.getBitWidth(), n);
    EXPECT_EQ(toInt.getBitWidth(), n);
    EXPECT_TRUE(origInt.eq(asInt));
    EXPECT_TRUE(origInt.eq(toInt));
  }
}

TEST(APLogicConstantTest, APUIntConstructorTest) {
  const uint64_t testVals[] = {0ULL, 1ULL, 5ULL, 235ULL, 923ULL,
   (355ULL << 16) + 45ULL, (854ULL << 24) + 111111ULL, ~(0ULL)};

  for (unsigned n: TEST_LENGTHS) {
    for (uint64_t tVal: testVals) {
      uint64_t refVal = tVal;
      if (n < 64)
        refVal &= (1ULL << n) - 1;
      auto apc = APLogicConstant(tVal, n);
      EXPECT_EQ(apc.getCodeLength(), LogicCodeLength::Log2);
      EXPECT_EQ(apc.getBitWidth(), n);
      EXPECT_EQ(refVal, apc.asAPInt().getZExtValue());
    }
  }
}

TEST(APLogicConstantTest, AllZeroTest) {
  for (auto n: TEST_LENGTHS) {
    auto zeroConst = APLogicConstant::getAllZeros(n);
    EXPECT_TRUE(zeroConst.isZero());
    EXPECT_TRUE(zeroConst.isZeroLike());
    auto apint = zeroConst.asAPInt();
    EXPECT_EQ(apint.getBitWidth(), n);
    EXPECT_TRUE(apint.isZero());
  }
}

TEST(APLogicConstantTest, isZeroLikeTest) {
  EXPECT_TRUE(APLogicConstant::fromChar<'0'>().isZero());
  EXPECT_TRUE(APLogicConstant::fromChar<'0'>().isZeroLike());

  EXPECT_FALSE(APLogicConstant::fromChar<'L'>().isZero());
  EXPECT_TRUE(APLogicConstant::fromChar<'L'>().isZeroLike());

  EXPECT_FALSE(APLogicConstant::fromChar<'1'>().isZero());
  EXPECT_FALSE(APLogicConstant::fromChar<'1'>().isZeroLike());
  EXPECT_FALSE(APLogicConstant::fromChar<'X'>().isZero());
  EXPECT_FALSE(APLogicConstant::fromChar<'X'>().isZeroLike());
  EXPECT_FALSE(APLogicConstant::fromChar<'Z'>().isZero());
  EXPECT_FALSE(APLogicConstant::fromChar<'Z'>().isZeroLike());
  EXPECT_FALSE(APLogicConstant::fromChar<'W'>().isZero());
  EXPECT_FALSE(APLogicConstant::fromChar<'W'>().isZeroLike());
  EXPECT_FALSE(APLogicConstant::fromChar<'U'>().isZero());
  EXPECT_FALSE(APLogicConstant::fromChar<'U'>().isZeroLike());
  EXPECT_FALSE(APLogicConstant::fromChar<'-'>().isZero());
  EXPECT_FALSE(APLogicConstant::fromChar<'-'>().isZeroLike());
  EXPECT_FALSE(APLogicConstant::fromChar<'H'>().isZero());
  EXPECT_FALSE(APLogicConstant::fromChar<'H'>().isZeroLike());

  auto str = std::string(250, '0');
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isZero());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isZeroLike());
  str[249] = 'L';
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isZero());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isZeroLike());
  str[75] = 'H';
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isZero());
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isZeroLike());
  str[249] = '0'; str[75] = '0'; str[1] = '1';
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isZero());
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isZeroLike());
  str[191] = 'L'; str[1] = '0';
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isZero());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isZeroLike());

  str = std::string(128, '0');
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isZero());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isZeroLike());
  str[127] = 'X';
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isZero());
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isZeroLike());
  str[127] = '0'; str[0] = 'L';
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isZero());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isZeroLike());
}

TEST(APLogicConstantTest, EqualsTest) {
  auto apcZero3 = APLogicConstant(3);
  EXPECT_TRUE(apcZero3 == apcZero3);
  auto apcZero2 = APLogicConstant(2);
  EXPECT_FALSE(apcZero3 == apcZero2);
  EXPECT_FALSE(apcZero2 == apcZero3);
  EXPECT_NE(hash_value(apcZero2), hash_value(apcZero3));

  auto apcAnotherZero3 = APLogicConstant(APLogic("000"));
  EXPECT_TRUE(apcAnotherZero3 == apcZero3);
  EXPECT_TRUE(apcZero3 == apcAnotherZero3);
  EXPECT_EQ(hash_value(apcZero3), hash_value(apcAnotherZero3));

  auto apcZero64 = APLogicConstant(0UL, 64);
  auto apcAnother64 = APLogicConstant(1UL << 32, 64);
  EXPECT_TRUE(apcZero64 == apcZero64);
  EXPECT_EQ(hash_value(apcZero64), hash_value(apcZero64));
  EXPECT_TRUE(apcAnother64 == apcAnother64);
  EXPECT_EQ(hash_value(apcAnother64), hash_value(apcAnother64));
  EXPECT_FALSE(apcZero64 == apcAnother64);
  EXPECT_FALSE(apcAnother64 == apcZero64);
  EXPECT_NE(hash_value(apcZero64), hash_value(apcAnother64));

  auto str128 = std::string(128, '0');
  str128[0] = '1'; str128[64] = '1';
  auto str64 = std::string(64, '0');
  str64[0] = 'X';
  auto apc11 = APLogicConstant(APLogic(str128));
  auto apcX = APLogicConstant(APLogic(str64));
  EXPECT_FALSE(apc11 == apcX);
  EXPECT_FALSE(apcX == apc11);
  EXPECT_NE(hash_value(apcX), hash_value(apc11));

  str128[62] = 'Z';
  auto apc1Z1 = APLogicConstant(APLogic(str128));
  EXPECT_FALSE(apc11 == apc1Z1);
  EXPECT_FALSE(apc1Z1 == apc11);
  EXPECT_NE(hash_value(apc11), hash_value(apc1Z1));
  
  auto apc1Z1Copy =  APLogicConstant(apc1Z1);
  EXPECT_TRUE(apc1Z1 == apc1Z1Copy);
  EXPECT_TRUE(apc1Z1Copy == apc1Z1);
  EXPECT_EQ(hash_value(apc1Z1Copy), hash_value(apc1Z1));

  str128[62] = 'U';
  auto apc1U1 = APLogicConstant(APLogic(str128));
  EXPECT_TRUE(apc1U1 == apc1U1);
  EXPECT_FALSE(apc1U1 == apc1Z1);
  EXPECT_FALSE(apc1Z1 == apc1U1);
  EXPECT_NE(hash_value(apc1U1), hash_value(apc1Z1));

  str128[126] = 'W';
  auto apcW1U1 = APLogicConstant(APLogic(str128));
  EXPECT_FALSE(apcW1U1 == apc1Z1);
  EXPECT_FALSE(apc1Z1 == apcW1U1);
  EXPECT_NE(hash_value(apc1Z1), hash_value(apcW1U1));
  EXPECT_FALSE(apcW1U1 == apc11);
  EXPECT_FALSE(apc11 == apcW1U1);
  EXPECT_NE(hash_value(apcW1U1), hash_value(apc11));

  for (unsigned i = 0; i < 9; i++) {
    for (unsigned j = 0; j < 9; j++) {
      APLogicConstant apcA = APLogicConstant(APLogic(16, L9CHARS[i]));
      APLogicConstant apcB = APLogicConstant(APLogic(16, L9CHARS[j]));
      if (i == j) {
        EXPECT_TRUE(apcA == apcB);
        EXPECT_EQ(hash_value(apcA), hash_value(apcB));
      } else {
        EXPECT_FALSE(apcA == apcB);
        EXPECT_NE(hash_value(apcA), hash_value(apcB));
      }
    }
  }

  APLogicConstant emptyKey = llvm::DenseMapInfo<circt::hw::APLogicConstant, void>::getEmptyKey();
  APLogicConstant tombstoneKey = llvm::DenseMapInfo<circt::hw::APLogicConstant, void>::getTombstoneKey();
  EXPECT_TRUE(emptyKey == emptyKey);
  EXPECT_TRUE(tombstoneKey == tombstoneKey);
  EXPECT_FALSE(emptyKey == tombstoneKey);
  EXPECT_FALSE(tombstoneKey == emptyKey);
}

TEST(APLogicConstantTest, isIntegerLikeTest) {
  EXPECT_TRUE(APLogicConstant::fromChar<'0'>().isInteger());
  EXPECT_TRUE(APLogicConstant::fromChar<'0'>().isIntegerLike());
  EXPECT_FALSE(APLogicConstant::fromChar<'0'>().containsUnknownValues());
  EXPECT_TRUE(APLogicConstant::fromChar<'1'>().isInteger());
  EXPECT_TRUE(APLogicConstant::fromChar<'1'>().isIntegerLike());
  EXPECT_FALSE(APLogicConstant::fromChar<'1'>().containsUnknownValues());

  EXPECT_FALSE(APLogicConstant::fromChar<'L'>().isInteger());
  EXPECT_TRUE(APLogicConstant::fromChar<'L'>().isIntegerLike());
  EXPECT_FALSE(APLogicConstant::fromChar<'L'>().containsUnknownValues());
  EXPECT_FALSE(APLogicConstant::fromChar<'H'>().isInteger());
  EXPECT_TRUE(APLogicConstant::fromChar<'H'>().isIntegerLike());
  EXPECT_FALSE(APLogicConstant::fromChar<'H'>().containsUnknownValues());

  EXPECT_FALSE(APLogicConstant::fromChar<'X'>().isInteger());
  EXPECT_FALSE(APLogicConstant::fromChar<'X'>().isIntegerLike());
  EXPECT_TRUE(APLogicConstant::fromChar<'X'>().containsUnknownValues());
  EXPECT_FALSE(APLogicConstant::fromChar<'Z'>().isInteger());
  EXPECT_FALSE(APLogicConstant::fromChar<'Z'>().isIntegerLike());
  EXPECT_TRUE(APLogicConstant::fromChar<'Z'>().containsUnknownValues());
  EXPECT_FALSE(APLogicConstant::fromChar<'W'>().isInteger());
  EXPECT_FALSE(APLogicConstant::fromChar<'W'>().isIntegerLike());
  EXPECT_TRUE(APLogicConstant::fromChar<'W'>().containsUnknownValues());
  EXPECT_FALSE(APLogicConstant::fromChar<'U'>().isInteger());
  EXPECT_FALSE(APLogicConstant::fromChar<'U'>().isIntegerLike());
  EXPECT_TRUE(APLogicConstant::fromChar<'U'>().containsUnknownValues());
  EXPECT_FALSE(APLogicConstant::fromChar<'-'>().isInteger());
  EXPECT_FALSE(APLogicConstant::fromChar<'-'>().isIntegerLike());
  EXPECT_TRUE(APLogicConstant::fromChar<'-'>().containsUnknownValues());

  auto str = std::string(250, '0');
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isInteger());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isIntegerLike());
  EXPECT_FALSE(APLogicConstant(APLogic(str)).containsUnknownValues());

  str[72] = 'X';
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isInteger());
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isIntegerLike());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).containsUnknownValues());

  str[72] = 'W';
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isInteger());
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isIntegerLike());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).containsUnknownValues());

  str[72] = '1';
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isInteger());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isIntegerLike());
  EXPECT_FALSE(APLogicConstant(APLogic(str)).containsUnknownValues());

  str[248] = 'H';
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isInteger());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isIntegerLike());
  EXPECT_FALSE(APLogicConstant(APLogic(str)).containsUnknownValues());

  str[4] = 'L';
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isInteger());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).isIntegerLike());
  EXPECT_FALSE(APLogicConstant(APLogic(str)).containsUnknownValues());

  str[4] = 'U';
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isInteger());
  EXPECT_FALSE(APLogicConstant(APLogic(str)).isIntegerLike());
  EXPECT_TRUE(APLogicConstant(APLogic(str)).containsUnknownValues());
}

TEST(APLogicTest, APLogicConstantReplaceTest) {
  APLogic destA = APLogic(60);
  replTest(destA, APLogicConstant::fromChar<'1'>(), 0, 1, 60, 0);
  replTest(destA, APLogicConstant::fromChar<'Z'>(), 1, 1, 59, 0);
  replTest(destA, APLogicConstant::fromChar<'U'>(), 2, 1, 58, 0);
  replTest(destA, APLogicConstant::fromChar<'0'>(), 23, 1, 12, 0);

  auto convConst = APLogicConstant(destA);
  EXPECT_EQ(convConst.getCodeLength(), LogicCodeLength::Log9);
  
  APLogic destB = APLogic(128);
  replTest(destB, APLogicConstant::fromChar<'1'>(), 3, 1, 60, 0);
  replTest(destB, APLogicConstant::fromChar<'X'>(), 10, 1, 60, 0);
  replTest(destB, APLogicConstant::fromChar<'-'>(), 40, 1, 60, 0);
  replTest(destB, convConst, 68, 60, 1, 0); 

  APLogic destC = APLogic(256);
  replTest(destC, APLogicConstant::fromChar<'W'>(), 64, 1, 128, 0);
  replTest(destC, APLogicConstant::fromChar<'Z'>(), 192, 1, 64, 0);
  replTest(destC, APLogicConstant::fromChar<'U'>(), 0, 1, 64, 0);

  APLogic apl64 = APLogic(64);
  for (unsigned n: TEST_LENGTHS) {
    if (n > 64)
      break;
    APLogicConstant apcl2 = APLogicConstant(APLogic(std::string(RANDOML2 + n, n)));
    APLogicConstant apcl4 = APLogicConstant(APLogic(std::string(RANDOML4 + n, n)));
    APLogicConstant apcl9 = APLogicConstant(APLogic(std::string(RANDOML9 + n, n)));
    replTest(apl64, apcl2, 0, n, 1, 0);
    replTest(apl64, apcl4, 64 - n, n, 1, 0);
    replTest(apl64, apcl9, 0, n, 1, 0);
  }

  APLogicConstant apcl2 = APLogicConstant(APLogic(std::string(RANDOML2 + 100, 64)));
  APLogicConstant apcl4 = APLogicConstant(APLogic(std::string(RANDOML4 + 100, 32)));
  APLogicConstant apcl9 = APLogicConstant(APLogic(std::string(RANDOML9 + 100, 16)));

  replTest(apl64, apcl2, 4, 40, 1, 12);
  replTest(apl64, apcl4, 30, 20, 1, 3);
  replTest(apl64, apcl9, 10, 13, 1, 1);

  replTest(apl64, apcl2, 2, 30, 2, 13);
  replTest(apl64, apcl4, 17, 5, 8, 4);
  replTest(apl64, apcl9, 52, 3, 4, 2);

}

} // namespace
