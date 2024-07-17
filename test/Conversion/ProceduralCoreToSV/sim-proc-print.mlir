// RUN: circt-opt -lower-procedural-core-to-sv %s | FileCheck %s

// CHECK-LABEL: hw.module @basic_print
// CHECK-NEXT:    %[[TRG:.+]] = seq.from_clock %clk
// CHECK-NEXT:    sv.always posedge %[[TRG]] {
// CHECK-NEXT:      %[[FD:.+]] = hw.constant -2147483646 : i32
// CHECK-NEXT:      sv.fwrite %[[FD]], "Test %% Test"
// CHECK-NEXT:      sv.fwrite %[[FD]], "%b"(%val) : i8
// CHECK-NEXT:      %[[UNS:.+]] = sv.system "unsigned"(%val) : (i8) -> i8
// CHECK-NEXT:      sv.fwrite %[[FD]], "%d"(%[[UNS]]) : i8
// CHECK-NEXT:      %[[SGN:.+]] = sv.system "signed"(%val) : (i8) -> i8
// CHECK-NEXT:      sv.fwrite %[[FD]], "%d"(%[[SGN]]) : i8
// CHECK-NEXT:      sv.fwrite %[[FD]], "%x"(%val) : i8
// CHECK-NEXT:      sv.fwrite %[[FD]], "%c"(%val) : i8
// CHECK-NEXT:    }

hw.module @basic_print(in %clk : !seq.clock, in %val: i8) {
    %0 = seq.from_clock %clk
    hw.triggered posedge %0(%val) : i8 {
    ^bb0(%arg0: i8):
      %lit = sim.fmt.lit "Test % Test"
      sim.proc.print %lit
      %b = sim.fmt.bin %arg0 : i8
      sim.proc.print %b
      %u = sim.fmt.dec %arg0 : i8
      sim.proc.print %u
      %s = sim.fmt.dec signed  %arg0 : i8
      sim.proc.print %s
      %h = sim.fmt.hex %arg0 : i8
      sim.proc.print %h
      %c = sim.fmt.char %arg0 : i8
      sim.proc.print %c
    }
  }

// CHECK-LABEL: hw.module @basic_condition
// CHECK-NEXT:    %[[TRG:.+]] = seq.from_clock %clk
// CHECK-NEXT:    sv.always posedge %[[TRG]] {
// CHECK-NEXT:      %[[FD:.+]] = hw.constant {{-?[0-9]+}} : i32
// CHECK-NEXT:      sv.if %ping {
// CHECK-NEXT:        sv.fwrite %[[FD]], "Ping\0A"
// CHECK-NEXT:      }
// CHECK-NEXT:    }

hw.module @basic_condition(in %clk : !seq.clock, in %ping: i1) {
  %0 = seq.from_clock %clk
  hw.triggered posedge %0(%ping) : i1 {
  ^bb0(%arg0: i1):
    %lit = sim.fmt.lit "Ping\0A"
    scf.if %arg0 {
      sim.proc.print %lit
    }
  }
}

// CHECK-LABEL: hw.module @multi_clock
// CHECK-DAG:     %[[TRGA:.+]] = seq.from_clock %clka
// CHECK-DAG:     %[[TRGB:.+]] = seq.from_clock %clkb
// CHECK-DAG:     %[[TRGC:.+]] = seq.from_clock %clkc
// CHECK-NEXT:    sv.always posedge %[[TRGA]] {
// CHECK-NEXT:      %[[FDA:.+]] = hw.constant {{-?[0-9]+}} : i32
// CHECK-NEXT:      sv.if %ping {
// CHECK-NEXT:        sv.fwrite %[[FDA]], "Ping A\0A"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.always posedge %[[TRGB]] {
// CHECK-NEXT:      %[[FDB:.+]] = hw.constant {{-?[0-9]+}} : i32
// CHECK-NEXT:      sv.fwrite %[[FDB]], "Ping B\0A"
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.always posedge %[[TRGC]] {
// CHECK-NEXT:      %[[FDC:.+]] = hw.constant {{-?[0-9]+}} : i32
// CHECK-NEXT:      sv.if %ping {
// CHECK-NEXT:        sv.fwrite %[[FDC]], "Ping C\0A"
// CHECK-NEXT:      }
// CHECK-NEXT:    }

hw.module @multi_clock(in %clka : !seq.clock, in %clkb : !seq.clock, in %clkc : !seq.clock, in %ping: i1) {
  %0 = seq.from_clock %clka
  %1 = seq.from_clock %clkb
  %2 = seq.from_clock %clkc
  hw.triggered posedge %0(%ping) : i1 {
  ^bb0(%arg0: i1):
    %lit = sim.fmt.lit "Ping A\0A"
    scf.if %arg0 {
      sim.proc.print %lit
    }
  }
  hw.triggered posedge %1 {
    %lit = sim.fmt.lit "Ping B\0A"
    sim.proc.print %lit
  }
  hw.triggered posedge %2(%ping) : i1 {
  ^bb0(%arg0: i1):
    %lit = sim.fmt.lit "Ping C\0A"
    scf.if %arg0 {
      sim.proc.print %lit
    }
  }
}

// CHECK-LABEL: hw.module @else_condition
// CHECK-NEXT:    %[[TRG:.+]] = seq.from_clock %clk
// CHECK-NEXT:    sv.always posedge %[[TRG]] {
// CHECK-NEXT:      %[[FD:.+]] = hw.constant {{-?[0-9]+}} : i32
// CHECK-NEXT:      sv.if %ping {
// CHECK-NEXT:        sv.fwrite %[[FD]], "Then"
// CHECK-NEXT:      } else {
// CHECK-NEXT:        sv.fwrite %[[FD]], "Else"
// CHECK-NEXT:      }
// CHECK-NEXT:    }

hw.module @else_condition(in %clk : !seq.clock, in %ping: i1) {
  %0 = seq.from_clock %clk
  hw.triggered posedge %0(%ping) : i1 {
  ^bb0(%arg0: i1):
    %then = sim.fmt.lit "Then"
    %else = sim.fmt.lit "Else"
    scf.if %arg0 {
      sim.proc.print %then
    } else {
      sim.proc.print %else
    }
  }
}

// CHECK-LABEL: hw.module @format_concat
// CHECK-NEXT:    %[[TRG:.+]] = seq.from_clock %clk
// CHECK-NEXT:    sv.always posedge %[[TRG]] {
// CHECK-NEXT:      %[[FD:.+]] = hw.constant {{-?[0-9]+}} : i32
// CHECK-NEXT:      %[[UNS:.+]] = sv.system "unsigned"(%val) : (i32) -> i32
// CHECK-NEXT:      sv.fwrite %[[FD]], "Bin: 0b%b Dec: %%%%\\%%%d Hex: 0x%x"(%val, %[[UNS]], %val) : i32, i32, i32
// CHECK-NEXT:    }

hw.module @format_concat(in %clk : !seq.clock, in %val: i32) {
  %0 = seq.from_clock %clk
  hw.triggered posedge %0(%val) : i32 {
  ^bb0(%arg0: i32):
    %lb = sim.fmt.lit "Bin: "
    %pb = sim.fmt.lit "0b"
    %ph = sim.fmt.lit "0x"
    %ld = sim.fmt.lit "Dec: "
    %lh = sim.fmt.lit "Hex: "
    %epsilon = sim.fmt.lit ""
    %space = sim.fmt.lit " "
    %perc = sim.fmt.lit "%%\\%"

    %fb = sim.fmt.bin %arg0 : i32
    %bin = sim.fmt.concat (%lb, %pb, %fb)
    %fd = sim.fmt.dec %arg0 : i32
    %dec = sim.fmt.concat (%ld, %perc, %fd)
    %fh = sim.fmt.hex %arg0 : i32
    %hex = sim.fmt.concat (%lh, %ph, %epsilon, %fh)

    %cat = sim.fmt.concat (%bin, %space, %dec, %space, %hex)
    sim.proc.print %cat
  }
}
