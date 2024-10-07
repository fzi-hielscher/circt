// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK-LABEL: Run:

// CHECK-NEXT:   1 - One!
// CHECK-NEXT:   2 - Two!!
// CHECK-NEXT:   3 - Three!!!
// CHECK-NEXT:   4 - Four!!!!
// CHECK-NEXT: Ah-ah-ah!

 llvm.mlir.global @callCounterVal(0 : i8) : i8

 func.func @callCounter() -> i8 {
    %ptr = llvm.mlir.addressof @callCounterVal : !llvm.ptr
    %val = llvm.load %ptr : !llvm.ptr -> i8
    %cst1 = llvm.mlir.constant (1 : i8) : i8
    %inc = llvm.add %val, %cst1 : i8
    llvm.store %inc, %ptr : i8, !llvm.ptr
    return %inc : i8
 }

hw.module private @TheCount(in %clock : !seq.clock, in %expected: i8, in %trigger: !sim.trigger.init, out pass : i1) {
    %c0_i8 = hw.constant 0 : i8
    %false = hw.constant false
    %true = hw.constant true

    %callCount = sim.triggered(%expected) on (%trigger : !sim.trigger.init) if %true tieoff [0 : i8] {
        ^bb0(%arg0 : i8):
        %cst1_i8 = hw.constant 1 : i8
        %cst2_i8 = hw.constant 2 : i8
        %cst3_i8 = hw.constant 3 : i8
        %cst4_i8 = hw.constant 4 : i8

        %ctr = func.call @callCounter() : () -> i8
        %fmt = sim.fmt.dec %ctr : i8
        sim.proc.print %fmt

        %expectedOne = comb.icmp bin eq %arg0, %cst1_i8 : i8
        scf.if %expectedOne {
            %oneLit = sim.fmt.lit " - One!\n"
            sim.proc.print %oneLit
        }
        %expectedTwo = comb.icmp bin eq %arg0, %cst2_i8 : i8
        scf.if %expectedTwo {
            %twoLit = sim.fmt.lit " - Two!!\n"
            sim.proc.print %twoLit
        }
        %expectedThree = comb.icmp bin eq %arg0, %cst3_i8 : i8
        scf.if %expectedThree {
            %threeLit = sim.fmt.lit " - Three!!!\n"
            sim.proc.print %threeLit
        }
        %expectedFour = comb.icmp bin eq %arg0, %cst4_i8 : i8
        scf.if %expectedFour {
            %fourLit = sim.fmt.lit " - Four!!!!\n"
            sim.proc.print %fourLit
        }

        sim.yield_seq %ctr : i8
    } : (i8) -> (i8)

    %match = comb.icmp bin eq %callCount, %expected : i8

    %initFalse = seq.initial () {
        %0 =  hw.constant false
        seq.yield %0 : i1
    } : () -> !seq.immutable<i1>
    %reg = seq.compreg %match, %clock reset %false, %false initial %initFalse : i1
    hw.output %reg : i1
}

hw.module @Top(in %clock : !seq.clock, in %test : i1)  {
    %true = hw.constant true
    %cst1_i8 = hw.constant 1 : i8
    %cst2_i8 = hw.constant 2 : i8
    %cst3_i8 = hw.constant 3 : i8
    %cst4_i8 = hw.constant 4 : i8

    %init = sim.on_init
    %trig:5 = sim.trigger_sequence %init, 5 : !sim.trigger.init

    %run = sim.fmt.lit "Run:\0A"
    sim.print %run on %trig#0 if %true : !sim.trigger.init

    %clockTrig = sim.on_edge posedge %clock
    %litPass = sim.fmt.lit "Ah-ah-ah!\0A"
    %printPass = comb.and bin %test, %pass1, %pass2, %pass3, %pass4 : i1
    sim.print %litPass on %clockTrig if %printPass : !sim.trigger.edge<posedge>

    %pass4 = hw.instance "four"  @TheCount(clock : %clock: !seq.clock, expected : %cst4_i8 : i8, trigger : %trig#4 : !sim.trigger.init) -> (pass : i1)
    %pass1 = hw.instance "two"   @TheCount(clock : %clock: !seq.clock, expected : %cst2_i8 : i8, trigger : %trig#2 : !sim.trigger.init) -> (pass : i1)
    %pass2 = hw.instance "one"   @TheCount(clock : %clock: !seq.clock, expected : %cst1_i8 : i8, trigger : %trig#1 : !sim.trigger.init) -> (pass : i1)
    %pass3 = hw.instance "three" @TheCount(clock : %clock: !seq.clock, expected : %cst3_i8 : i8, trigger : %trig#3 : !sim.trigger.init) -> (pass : i1)

    hw.output
}

func.func @main() {
  %zero = arith.constant false
  %one = arith.constant  true
  %high = seq.const_clock high
  %low = seq.const_clock low

  arc.sim.instantiate @Top as %model {
    arc.sim.set_input %model, "test" = %zero : i1, !arc.sim.instance<@Top>
    arc.sim.set_input %model, "clock" = %low : !seq.clock, !arc.sim.instance<@Top>
    arc.sim.step %model : !arc.sim.instance<@Top>
    arc.sim.set_input %model, "clock" = %high : !seq.clock, !arc.sim.instance<@Top>
    arc.sim.step %model : !arc.sim.instance<@Top>

    arc.sim.set_input %model, "test" = %one : i1, !arc.sim.instance<@Top>
    arc.sim.set_input %model, "clock" = %low : !seq.clock, !arc.sim.instance<@Top>
    arc.sim.step %model : !arc.sim.instance<@Top>
    arc.sim.set_input %model, "clock" = %high : !seq.clock, !arc.sim.instance<@Top>
    arc.sim.step %model : !arc.sim.instance<@Top>
  }

  return
}
