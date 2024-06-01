// RUN: arcilator %s --run --jit-entry=main 2>&1 >/dev/null | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK: - dead % dead - FI%% (0000) BU%% ( 0 \  0)
// CHECK: FI%% (0000) BU%% ( 0 \  0)
// CHECK: 1
// CHECK: 2
// CHECK: FI%% (0011)
// CHECK: 4
// CHECK: BU%% ( 5 \  5)
// CHECK: FI%% (0110)
// CHECK: 7
// CHECK: 8
// CHECK: FI%% (1001)
// CHECK: BU%% (10 \ -6)
// CHECK: b
// CHECK: FI%% (1100)
// CHECK: d
// CHECK: e
// CHECK: - dead % dead - FI%% (1111) BU%% (15 \ -1)
  
sv.macro.decl @SYNTHESIS
  sv.macro.decl @PRINTF_COND
  sv.macro.decl @PRINTF_COND_
  emit.fragment @PRINTF_COND_FRAGMENT {
    sv.verbatim "\0A// Users can define 'PRINTF_COND' to add an extra gate to prints."
    sv.ifdef  @PRINTF_COND_ {
    } else {
      sv.ifdef  @PRINTF_COND {
        sv.macro.def @PRINTF_COND_ "(`PRINTF_COND)"
      } else {
        sv.macro.def @PRINTF_COND_ "1"
      }
    }
  }
  hw.module @Example(in %clock : !seq.clock, in %reset : i1, in %en : i1) attributes {emit.fragments = [@PRINTF_COND_FRAGMENT]} {
    %c1_i4 = hw.constant 1 : i4
    %c5_i4 = hw.constant 5 : i4
    %c3_i4 = hw.constant 3 : i4
    %litDash = sim.fmt.lit "-"
    %litPerc = sim.fmt.lit "%"
    %litSpace = sim.fmt.lit " "
    %cst57005 = hw.constant 57005 : i16
    %cstHex = sim.fmt.hex %cst57005 : i16
    %cat = sim.fmt.concat %litSpace, %cstHex, %litSpace
    %catcat = sim.fmt.concat %litDash, %cat, %litPerc, %cat, %litDash, %litSpace
    %1 = sim.fmt.lit " \\ "
    %2 = sim.fmt.lit "BU%% ("
    %3 = sim.fmt.lit ") "
    %4 = sim.fmt.lit "FI%% ("
    %5 = sim.fmt.lit "\0A"
    %c0_i3 = hw.constant 0 : i3
    %c0_i2 = hw.constant 0 : i2
    %c0_i4 = hw.constant 0 : i4
    %true = hw.constant true
    %foo = seq.firreg %7 clock %clock reset sync %reset, %c0_i4 {firrtl.random_init_start = 0 : ui64} : i4
    %6 = comb.add %foo, %c1_i4 {sv.namehint = "_foo_T"} : i4
    %7 = comb.mux bin %en, %6, %foo : i4
    %8 = comb.xor bin %en, %true : i1
    %9 = comb.xor bin %reset, %true : i1
    %10 = comb.and bin %8, %9 : i1
    %11 = comb.modu bin %foo, %c3_i4 : i4
    %12 = comb.extract %11 from 0 : (i4) -> i2
    %13 = comb.icmp bin ne %12, %c0_i2 : i2
    %14 = comb.modu bin %foo, %c5_i4 : i4
    %15 = comb.extract %14 from 0 : (i4) -> i3
    %16 = comb.icmp bin ne %15, %c0_i3 : i3
    %17 = comb.and bin %13, %16 : i1
    %18 = comb.and bin %17, %9 : i1
    %19 = sim.fmt.hex %foo : i4
    %20 = sim.fmt.concat %19, %5
    %21 = comb.xor bin %17, %true : i1
    %22 = comb.xor bin %13, %true : i1
    %23 = comb.and bin %21, %22 : i1
    %24 = comb.and bin %23, %9 : i1
    %25 = sim.fmt.bin %foo : i4
    %26 = sim.fmt.concat %4, %25, %3
    %27 = comb.xor bin %16, %true : i1
    %28 = comb.and bin %21, %27 : i1
    %29 = comb.and bin %28, %9 : i1
    %30 = sim.fmt.dec %foo : i4
    %31 = sim.fmt.dec signed %foo : i4
    %32 = sim.fmt.concat %2, %30, %1, %31, %3
    %33 = comb.and bin %21, %9 : i1
    sv.ifdef  @SYNTHESIS {
    } else {
      %PRINTF_COND_ = sv.macro.ref @PRINTF_COND_() : () -> i1
      %34 = comb.and bin %PRINTF_COND_, %10 : i1
      sim.printf %catcat on %clock if %34
      %35 = comb.and bin %PRINTF_COND_, %18 : i1
      sim.printf %20 on %clock if %35
      %36 = comb.and bin %PRINTF_COND_, %24 : i1
      sim.printf %26 on %clock if %36
      %37 = comb.and bin %PRINTF_COND_, %29 : i1
      sim.printf %32 on %clock if %37
      %38 = comb.and bin %PRINTF_COND_, %33 : i1
      sim.printf %5 on %clock if %38
    }
    hw.output
  }
  om.class @Example_Class(%basepath: !om.basepath) {
  }

func.func @main() {
  %zero = arith.constant false
  %one = arith.constant  true
  %high = seq.const_clock high
  %low = seq.const_clock low

  arc.sim.instantiate @Example as %model {
    arc.sim.set_input %model, "reset" = %one : i1, !arc.sim.instance<@Example>
    arc.sim.set_input %model, "en" = %zero : i1, !arc.sim.instance<@Example>

    arc.sim.set_input %model, "clock" = %low : !seq.clock, !arc.sim.instance<@Example>
    arc.sim.step %model : !arc.sim.instance<@Example>
    arc.sim.set_input %model, "clock" = %high : !seq.clock, !arc.sim.instance<@Example>
    arc.sim.step %model : !arc.sim.instance<@Example>

    arc.sim.set_input %model, "clock" = %low : !seq.clock, !arc.sim.instance<@Example>
    arc.sim.step %model : !arc.sim.instance<@Example>
    arc.sim.set_input %model, "clock" = %high : !seq.clock, !arc.sim.instance<@Example>
    arc.sim.step %model : !arc.sim.instance<@Example>

    arc.sim.set_input %model, "reset" = %zero : i1, !arc.sim.instance<@Example>

    arc.sim.set_input %model, "clock" = %low : !seq.clock, !arc.sim.instance<@Example>
    arc.sim.step %model : !arc.sim.instance<@Example>
    arc.sim.set_input %model, "clock" = %high : !seq.clock, !arc.sim.instance<@Example>
    arc.sim.step %model : !arc.sim.instance<@Example>

    arc.sim.set_input %model, "en" = %one : i1, !arc.sim.instance<@Example>

    %lb = arith.constant 0 : index
    %ub = arith.constant 15 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
        arc.sim.set_input %model, "clock" = %low : !seq.clock, !arc.sim.instance<@Example>
        arc.sim.step %model : !arc.sim.instance<@Example>
        arc.sim.set_input %model, "clock" = %high : !seq.clock, !arc.sim.instance<@Example>
        arc.sim.step %model : !arc.sim.instance<@Example>
    }

    arc.sim.set_input %model, "en" = %zero : i1, !arc.sim.instance<@Example>

    arc.sim.set_input %model, "clock" = %low : !seq.clock, !arc.sim.instance<@Example>
    arc.sim.step %model : !arc.sim.instance<@Example>
    arc.sim.set_input %model, "clock" = %high : !seq.clock, !arc.sim.instance<@Example>
    arc.sim.step %model : !arc.sim.instance<@Example>
  }

  return
}