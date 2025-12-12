//===- JITBind.h - ArcRuntime JIT symbol binding helper -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the helper struct used to bind the IR interface to the MLIR
// Execution Engine without relying on the linker.
//
// This file is specific to the runtime implementation statically linked into
// the arcilator tool and not part of the runtime's API.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_RUNTIME_JITBIND_H
#define ARC_RUNTIME_JITBIND_H

#include "ArcRuntime/Common.h"

namespace circt {
namespace arc {
namespace runtime {

struct APICallbacks {
  static const char *symName_allocInstance;
  uint8_t *(*allocInstance)(const struct ArcRuntimeModelInfo *model,
                            const char *args);
  static const char *symName_deleteInstance;
  void (*deleteInstance)(uint8_t *simState);
};

const APICallbacks &getArcRuntimeAPICallbacks();

} // namespace runtime
} // namespace arc
} // namespace circt

#endif // ARC_RUNTIME_JITBIND_H
