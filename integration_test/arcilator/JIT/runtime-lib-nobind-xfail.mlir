// RUN: arcilator %s --run --jit-entry=main --jit-no-runtime-bind
// REQUIRES: arcilator-jit
// XFAIL: *

// Check that the MLIR Execution Engine _cannot_ find the statically linked
// runtime library unless we explicitly bind it.
//
// Passing this test despite using `--jit-no-runtime-bind` without providing
// an alternative implementation indicates that the linked symbols are
// unintendedly exported from the arcilator executable.

hw.module @flipflop(in %clk: i1, in %d: i8, out q: i8) {
  %seq_clk = seq.to_clock %clk
  %reg = seq.compreg %d, %seq_clk : i8
  hw.output %reg : i8
}

func.func @main() {
  arc.sim.instantiate @flipflop as %model {
    %d = arc.sim.get_port %model, "d" : i8, !arc.sim.instance<@flipflop>
    arc.sim.emit "d", %d : i8
  }
  return
}
