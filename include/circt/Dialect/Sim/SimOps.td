//===- SimOps.td - `sim` dialect ops -----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for `sim`.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMOPS_TD
#define CIRCT_DIALECT_SIM_SIMOPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "circt/Dialect/Sim/SimDialect.td"
include "circt/Dialect/Sim/SimTypes.td"
include "circt/Dialect/Seq/SeqTypes.td"

class SimOp<string mnemonic, list<Trait> traits = []> :
    Op<SimDialect, mnemonic, traits>;

def PlusArgsTestOp : SimOp<"plusargs.test", [Pure]> {
  let summary = "SystemVerilog `$test$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs I1:$found);
  let assemblyFormat = "$formatString attr-dict";
}

def PlusArgsValueOp : SimOp<"plusargs.value", [Pure]> {
  let summary = "SystemVerilog `$value$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs I1:$found, AnyType:$result);
  let assemblyFormat = "$formatString attr-dict `:` type($result)";
}

def FinishOp : SimOp<"finish"> {
  let summary = "Simulation finish condition";

  let arguments = (ins ClockType:$clk, I1:$cond);
  let results = (outs);

  let assemblyFormat = "$clk `,` $cond attr-dict";
}

def FatalOp : SimOp<"fatal"> {
  let summary = "Simulation failure condition";

  let arguments = (ins ClockType:$clk, I1:$cond);
  let results = (outs);

  let assemblyFormat = "$clk `,` $cond attr-dict";
}


def FormatLitOp : SimOp<"fmt.lit", [Pure, ConstantLike]> {
  let summary = "String literal token for fomatted printing";

  let arguments = (ins StrAttr:$literal);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;
  let hasVerifier = true;

  let assemblyFormat = "$literal attr-dict";
}

def FormatHexOp : SimOp<"fmt.hex", [Pure]> {
  let summary = "Format integer value as (lower case) hexadecimal string";

  let arguments = (ins AnyInteger:$value);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = "$value attr-dict `:` qualified(type($value))";
}

def FormatDecOp : SimOp<"fmt.dec", [Pure]> {
  let summary = "Format integer value as decimal string";

  let arguments = (ins AnyInteger:$value, UnitAttr:$isSigned);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = "(`signed` $isSigned^)? $value attr-dict `:` qualified(type($value))";

  let extraClassDeclaration = [{
      static inline unsigned getDecimalWidth(unsigned bits, bool isSigned) {
        if (bits == 0)
          return 1;
        if (bits == 1)
          return isSigned ? 2 : 1;

        if (isSigned)
          bits--;

        // Should be precise up until bits = 13301
        const double baseConversionFactor = 0.30103; // log(2) / log(10) + epsilon
        unsigned digits = std::ceil(bits * baseConversionFactor);

        return isSigned ? digits + 1 : digits;
      }
  }];
}

def FormatBinOp : SimOp<"fmt.bin", [Pure]> {
  let summary = "Format integer value as binary string";

  let arguments = (ins AnyInteger:$value);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = "$value attr-dict `:` qualified(type($value))";
}

def FormatCharOp : SimOp<"fmt.char", [Pure]> {
  let summary = "Format integer value single ASCII character token";

  let arguments = (ins AnyInteger:$value);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = "$value attr-dict `:` qualified(type($value))";
}

def FormatStringConcatOp : SimOp<"fmt.concat", [Pure]> {
  let summary = "Concatenate formatted sting tokens";

  let arguments = (ins Variadic<FormatStringType>:$inputs);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;
  let hasCanonicalizeMethod = true;

  let assemblyFormat = "$inputs attr-dict";
}

def PrintFormattedOp : SimOp<"printf"> {
  let summary = "Print a formatted string on a given clock and condition";
  let arguments = (ins ClockType:$clock, FormatStringType:$input, Optional<I1>:$condition);
  let assemblyFormat = "$input `on` $clock  (`if` $condition^)? attr-dict";
  let hasCanonicalizeMethod = true;
}

def PrintFormattedProcOp : SimOp<"proc.printf"> {
  let summary = "Print a formatted string within a procedural region";
  let arguments = (ins Variadic<FormatStringType>:$inputs);
  let assemblyFormat = "$inputs attr-dict";
  let hasVerifier = true;
}

#endif // CIRCT_DIALECT_SIM_SIMOPS_TD
