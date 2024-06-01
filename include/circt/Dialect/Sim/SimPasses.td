#ifndef CIRCT_DIALECT_SIM_SEQPASSES
#define CIRCT_DIALECT_SIM_SEQPASSES

include "mlir/Pass/PassBase.td"

def ProceduralizeSim : Pass<"proceduralize-sim", "hw::HWModuleOp"> {
  let summary = "Create procedural regions for simulation operations";
  let dependentDialects = ["circt::hw::HWDialect, circt::seq::SeqDialect, mlir::scf::SCFDialect"];
}

def LowerPrintsToSV : Pass<"sim-lower-prints-to-sv", "hw::HWModuleOp"> {
  let summary = "Lower procedural printf regions to SystemVerilog";
  let dependentDialects = ["circt::sv::SVDialect"];
}

#endif // CIRCT_DIALECT_SIM_SEQPASSES