//===- SMTTypes.td - SMT dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMTYPES_TD
#define CIRCT_DIALECT_SIM_SIMTYPES_TD

include "circt/Dialect/Sim/SimDialect.td"
include "mlir/IR/AttrTypeBase.td"

class SimTypeDef<string name> : TypeDef<SimDialect, name> { }



def FormatStringType : SimTypeDef<"FormatString"> {
  let mnemonic = "fmtstr";
}


#endif // CIRCT_DIALECT_SIM_SIMTYPES_TD
