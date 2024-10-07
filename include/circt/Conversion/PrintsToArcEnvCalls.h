//===- ArcToLLVM.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_PRINTSTOARCENVCALLS_H
#define CIRCT_CONVERSION_PRINTSTOARCENVCALLS_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {
#define GEN_PASS_DECL_LOWERPRINTSTOARCENVCALLS
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createLowerPritnsToArcEnvCalls();
} // namespace circt

#endif // CIRCT_CONVERSION_PRINTSTOARCENVCALLS_H
