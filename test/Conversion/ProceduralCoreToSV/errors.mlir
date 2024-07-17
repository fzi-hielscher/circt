// RUN: circt-opt -lower-procedural-core-to-sv -split-input-file -verify-diagnostics %s

hw.module @fmt_arg(in %clk : !seq.clock, in %fmt: !sim.fstring) {
  %0 = seq.from_clock %clk
  hw.triggered posedge %0(%fmt): !sim.fstring {
  ^bb0(%arg0: !sim.fstring):
    // expected-error @below {{Format strings passed as block argument cannot be lowered.}}
    sim.proc.print %arg0
  }
}

// -----

hw.module @lit_if(in %clk : !seq.clock, in %cond: i1) {
  %0 = seq.from_clock %clk
  hw.triggered posedge %0(%cond): i1 {
  ^bb0(%arg0: i1):
    %foo = sim.fmt.lit "Foo"
    %bar = sim.fmt.lit "Bar"
    // expected-error @below {{SCF If operation with results cannot be converted to SV.}}
    // expected-error @below {{Unsupported format string operation.}}
    %lit = scf.if %arg0 -> !sim.fstring  {
      scf.yield %foo : !sim.fstring
    } else {
      scf.yield %bar : !sim.fstring
    }
    // expected-error @below {{Failed to lower format string.}}
    sim.proc.print %lit
  }
}

