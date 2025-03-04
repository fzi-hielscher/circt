# MVL Dialect Rationale

## Motivation and Scope

In both (System)Verilog and VHDL it is common practice to use _logic_ types and operations, that go beyond basic 0/1 boolean logic. In Verilog, 4-valued (0,1,X,Z) logic type is the quasi default for nets and variables. VHDL even uses nine different logic values. The Many Valued Logic (MVL) dialect aims to provide a unified representation of these systems, to be used in the middle-end of the CIRCT compiler framework. As such, it tries to be as front- and back-end agnostic as possible. E.g., it should be possible to parse a Verilog source to MVL and then re-emit it as behaviorally equivalent VHDL. To do so, it needs to be able to accurately model bespoke operation semantics found in these HDLs (e.g., the Verilog trinary operator ```a ? b : c``` or the VHDL-2008 ```a ?= b``` comparison).

The MVL Dialect can be used to describe _combinational_ logic as directed _dataflow_. The following aspects are **NOT** within the scope of this dialect:

* If you don't care about _strange_ values and are looking for more synthesis rather than simulation focused combinational logic, the ```comb``` dialect is the right place to go.
* A $\mathrm{Z}$ in MVL is merely a distinct value and does not imply any sort of bi-directionality / tristate IO. Checkout the ```hw.HiZ``` type if you need this.
* There are no stateful elements (i.e., registers or memories) in MVL. That's what the ```seq``` dialect is for.
* There is no concept of time in MVL. This is deferred to the ```llhd``` dialect.
* The operations in MVL have no side-effects or any implications on control-flow. If you need to steer your simulation, have a look at the  ```sim``` dialect.

## Types

The MVL dialect defines and operates on a single type: ```!mvl.logic``` \
```!mvl.logic``` can be used to model any HDL types that are compatible with the 9-valued logic defined in IEEE 1164. This includes the Verilog 4-valued ```logic``` (IEEE 1364) type, as well as VHDL ```std_logic``` and ```signed```/```unsigned``` types of the ```numeric_std``` package (IEEE 1164). ```!mvl.logic``` itself is signless. Any sign semantics have to be specified by the operations.\
```!mvl.logic``` always carries an integer-valued width attribute specifying the number of "digits" contained in the given type. The concrete width may or may not be known at compile time. E.g., both ```!mvl.logic<8>``` and ```!mvl.logic<#hw.param.decl.ref<"WIDTH">>``` are possible. Since parameter expressions are not constrained to positive values, the ```!mvl.logic``` type is also defined over zero and negative widths.

### Logic Vectors
An ```!mvl.logic``` value of _positive_ width $n$ is an $n$-element vector over the set of 9-valued logic digits:
```math
B := \lbrace \mathrm{0},\mathrm{1},\mathrm{H},\mathrm{L},\mathrm{U},\mathrm{W},\mathrm{X},\mathrm{Z},\mathrm{-} \rbrace
```

These digits _do not_ carry any implicit semantics. I.e., an $\mathrm{X}$ is just an $\mathrm{X}$ and distinct from the other eight digits. In the context of the MVL dialect, it does not imply any "unknown value" semantics.

### Empty Vector
If the width of a ```!mvl.logic``` type evaluates to a value less or equal zero, it can carry the special _empty vector_ value $\varepsilon$.

### Verilog and VHDL compatibility

MVL Logic vectors are always _normalized_. I.e. they start at index zero and grow towards positive infinity. For a logic vector of positive width $n \in \mathrm{N^+}$, the most significant digit is at index $n - 1$. This is in contrast to both VHDL and Verilog vectors, which are defined over ranges.

For VHDL, both  ```std_logic_vector(N downto M)``` as well as  ```std_logic_vector(M to N)``` types correspond to a ```!mvl.logic<N-M+1>``` type. For $N < M$ these types produce the _null range_ in VHDL. This is represented by $\varepsilon$ in MVL.

Mapping (System)Verilog is not as trivial, as both ranges ```[1:-1]``` and ```[-1:1]```  are valid and produce a vector of size three. An accurate representation of a ```logic [N:M]``` type is given by ```!mvl.logic<|N-M|+1>```. Notably, there is no way of creating an empty vector. The Abstract Upper Limit Attribute (see below) provides a mechanism to effectively exclude empty vectors from the MVL dialect's semantics. This is used to avoid the necessity of handling  non-positive width logic vectors separately when converting from MVL to SystemVerilog. It is then within the responsibility of the frontend to ensure that widths, parameterized or not, do not "underflow".

## Operations

### Constant
```mvl.constant``` produces a constant 0/1-only logic vector of known, positive width from an integer attribute. E.g.:

```%cst0101 = mvl.constant 5 : !mvl.logic<4> ```

### Literal
```mvl.literal``` can be used to produce arbitrary constant values of ```!mvl.logic``` type from a string attribute. In the simplest case the string directly matches the value of the logic vector:

```%cstUWU = mvl.literal "UWU" :  !mvl.logic<3>```\
```%cstEmpty = mvl.literal "" :  !mvl.logic<0>```

An ellipsis (```...```) in front of the string can be used to fill up the most significant digits with the first specified character (think: sign extension). The string must not be empty in this case.

```%cst64Xs = mvl.literal ..."X" :  !mvl.logic<64>```\
```%cst111ZX = mvl.literal ..."1ZX" :  !mvl.logic<5>```

If the width of the vector is not a known constant, use of the ellipsis is mandatory. The following example produces a logic vector of parameterized width, entirely filled with ones, or $\varepsilon$ iff ```WIDTH``` evaluates to a non-positive value:

```%cstOnes = mvl.literal ..."1" : !mvl.logic<#hw.param.decl.ref<"WIDTH">>```


### Elementwise AND/OR/XOR

The fundamental logic operations are derived from the truth tables as given by the IEEE 1164 standard.
The operations take a variadic list of equally sized logic vectors and produce a result of same type.
The truth-table is applied digit-by-digit recursively over all operands. \
If less than two operands are given, the list of operands is implicitly padded with constant vectors:
Let $k \in \mathbb{Z}$  be the width of the operation's result type. The ```mvl.or``` and  ```mvl.xor``` operations are padded with:\
```%zeros = mvl.literal ..."0" : !mvl.logic<k>```

``mvl.and`` is padded with: \
```%ones = mvl.literal ..."1" : !mvl.logic<k>```

Note that all three operations are both associative and commutative, so the order of operands is irrelevant.

### AND/OR/XOR Reduce

The reduction operations take a single logic vector operand and produce a result of type ```!mvl.logic<1>```.
The result of the reduction operations is defined to be equal to their elementwise counterparts where the input
vector is "unrolled" to single digit operands. \
I.e.: $\mathrm{ANDReduce}(v_{k-1} ... v_{1}v_{0}) = AND(v_{0}, v_{1}, ..., v_{k})$ \
If the width of the input vector is non-positive, the result is equal to the elementwise counterpart with no operand given. \
I.e., due to operand padding:
$\mathrm{ANDReduce}(\varepsilon)=AND(\mathrm{1},\mathrm{1})=\mathrm{1}$ \
Also:
$\mathrm{ORReduce}(\mathrm{Z})=OR(\mathrm{Z},\mathrm{0})=\mathrm{X}$

### TO_X01, TO_XZ01, TO_UX01

The ```mvl.to_x01```, ```mvl.to_xz01``` and ```mvl.to_ux01``` operations are defined to be equal to their IEEE 1164 counterparts.

### Elementwise Equals

The ```mvl.eltwise_eq``` operation performs a digit-by-digit comparison of two equally sized logic vectors.
The result vector has the same type as the operands, and contains a $\mathrm{1}$ at position $i$, if and only if
both operands contain _the same_ digit at their respective $i$-th position.
Otherwise, the result at position $i$ is $\mathrm{0}$.  E.g., the comparison of "1HXX" and "11ZX" results in "1001".
For empty vector operands the result is $\varepsilon$. \
```mvl.eltwise_eq``` stands out as the only operation that can dynamically discriminate between all nine logic digits.
Most other operations treat "similar" digits like $\mathrm{1}$ / $\mathrm{H}$ or $\mathrm{Z}$ / $\mathrm{X}$ as equivalent.


### Sign Extend or Truncate

The ```mvl.sext_or_trunc``` operation resizes a logic vector to a new width. Let $N$ be the width of the input and $M$ the width of the result. \
If $M \leq 0$ the result is $\varepsilon $, if $N = M$ the operand and result value are identical. \
If $N \leq 0 < M$ the result is equal to ```mvl.literal ..."0" : !mvl.logic<M>```. \
If $0 < M < N$ the most significant $N - M$ digits are truncated from the operand. \
If $0 < N < M$ the operand is padded (sign extended) by repeating the most significant digit $M - N$ times.

### Insert

The ```mvl.dyn_insert``` operation inserts a source logic vector into a destination logic vector at an (potentially negative) integer offset.
The type of the result is identical to the type of the destination operand.
If the width of the destination operand is non-positive, the result is $\varepsilon$\.
An attribute controls whether the offset operand is interpreted as (unsigned) positive or negative value. Let $k \in \mathbb{Z}$ be the value of the offset and $s \in \mathbb{Z} \: ,d \in \mathbb{N^+}$ the width of the source and destination operand respectively.
For the indices $i$ with $0 \leq i < d$ , the elements of the result vector are given by:
```math
result_i = \begin{cases}
    source_{i - k} & \text{if } \: 0 \leq i - k  < s \\
    dest_i & \text{otherwise}
\end{cases}
```

The ```mvl.insert``` operation is defined in the same way, but takes a constant _signed 32-bit integer_ offset attribute instead of a dynamic value.

### Casts

#### From Integer

```mvl.from_integer``` converts a value of built-in integer or ```hw.int``` type to a logic vector of matching width. ```i0``` values can be cast to any fixed non-positive width.

#### To Integer

Given a non-poison logic vector, containing only $\mathrm{0}$, $\mathrm{1}$, $\mathrm{L}$, and $\mathrm{H}$ digits, ```mvl.to_integer``` returns a corresponding integer value
of either built-in integer or ```hw.int``` type. $\mathrm{L}$ is interpreted as $\mathrm{0}$, $\mathrm{H}$ as $\mathrm{1}$.
If the width is known to be zero or negative, the result type is ```i0```.

If the input is $\mathrm{Poison}$ or contains any invalid digits, the resulting value is _undefined_.


The cast operations do _not_ assume values of type integer to strictly be integers. It is legal to substitute

```
%a = mvl.to_integer %foo : (!mvl.logic<32>) -> i32
%b = mvl.from_integer %a : (i32) -> !mvl.logic<32>
```
with the identity of ```%foo```. Other than that, no assumptions on the result of ```mvl.from_integer``` shall be made unless specified explicitly by an [AULA](#abstract-upper-limit-attribute). E.g.:

```
%reg = seq.compreg %d, %clk : i1
%mvl = mvl.from_integer [b0,b1,X] %reg : (i1) -> !mvl.logic<1>
```

permits the register's uninitialized state to be modeled by an $\mathrm{X}$ in simulation, but implies that it will never become $\mathrm{U}$ or $\mathrm{Z}$.

### Select

The ```mvl.select``` operation chooses between two possible values of same type based on a selector operand of type ```mvl.logic<1>```.
If the selector is $\mathrm{0}$ the "false" operand is returned. if the selector is $\mathrm{1}$ the "true" operand is returned. Notably, if the de-selected operand is $\mathrm{Poison}$, the $\mathrm{Poison}$ value is _not_ propagated. On the other hand, if the selector is neither $\mathrm{0}$ nor $\mathrm{1}$, the result is $\mathrm{Poison}$, irrespective of the other operands.


### Integer Or Else

Arithmetic operations follow a common pattern in Verilog and VHDL: If any operand contains an "unknown" digit at any position, the result of the operation is all X. Other than that, they follow conventional integer arithmetic. Since integer arithmetic is already implemented in the ```comb``` dialect, MVL only provides a generic wrapper operation. E.g., an addition can be described as follows:

```MLIR
%x = mvl.literal ..."X" : !mvl.logic<16>
%sum = mvl.int_or_else (%a, %b : !mvl.logic<16>, !mvl.logic<16>) else %x : !mvl.logic<16> {
    ^bb0(%arg0: i16, %arg1: i16):
    %0 = comb.add %arg0, %arg1 : i16
    mvl.yield %0 : i16
}
```

```mvl.int_or_else``` takes a variadic list of logic vectors, which are converted to the body's integer arguments following the semantics of ```mvl.to_int```. The ```mvl.yield``` terminator converts its single integer operand to a logic vector following the semantics of ```mvl.from_integer```. If all of the digits of all of the operation's operands are within the set $\{ \mathrm{0},\mathrm{1},\mathrm{L},\mathrm{H} \}$, the result of the operation is the converted yielded value. Otherwise the result is equal to the provided 'else' operand. \
The example above is logically equivalent to:

```MLIR
%x    = mvl.literal ..."X" : !mvl.logic<16>
%atox = mvl.to_x01 %a : !mvl.logic<16>
%btox = mvl.to_x01 %b : !mvl.logic<16>
%eeqa = mvl.eltwise_eq %atox, %x : !mvl.logic<16>
%eeqb = mvl.eltwise_eq %btox, %x : !mvl.logic<16>
%or   = mvl.or %eeqa, %eeqb : !mvl.logic<16>
%anyx = mvl.or_reduce %or : !mvl.logic<1>
%arg0 = mvl.to_int %a : (!mvl.logic<16>) -> i16
%arg1 = mvl.to_int %b : (!mvl.logic<16>) -> i16
%0    = comb.add %arg0, %arg1 : i16
%conv = mvl.from_int %0 : (i16) -> !mvl.logic<16>
%sum  = mvl.select %anyx, %x, %conv : !mvl.logic<16>
```

### Match

The ```mvl.match``` operation implements logical equality of two logic vectors of same size, with an optional mask operand of the same type.

```
%result = mvl.match %a, %b, %mask : !mvl.logic<N>
```

is _defined_ to be equal to

```
%ones   = mvl.literal ..."1" : !mvl.logic<N>
%xnor   = mvl.xor %a, %b, %ones : !mvl.logic<N>
%masked = mvl.or %xnor, %mask : !mvl.logic<N>
%result = mvl.and_reduce %masked : !mvl.logic<N>
```

If the mask operand is not explicitly provided, it is an all-zero vector.

### Mux

The ```mvl.mux``` implements a selection that is resolved digit by digit.

```MLIR
%out = mvl.mux %s, %a, %b : !mvl.logic<N>
```

is _defined_ to be equal to

```
%ones = mvl.literal ..."1" : !mvl.logic<N>
%sext = mvl.sext_or_trunc %s : (!mvl.logic<1>) -> !mvl.logic<N>
%nots = mvl.xor %sext, %ones : !mvl.logic<N>
%t0   = mvl.and %a, %sext : !mvl.logic<N>
%t1   = mvl.and %b, %nots : !mvl.logic<N>
%t2   = mvl.and %a, %b : !mvl.logic<N>
%out  = mvl.or %t0, %t1, %t2 : !mvl.logic<N>
```

In contrast to ```mvl.select``` the Mux allows the selector operand to become $\mathrm{X}$ without producing $\mathrm{Poison}$. On the other hand, a  $\mathrm{Poison}$ value on the input operands is propagated unconditionally. ```mvl.mux``` can also _not_ be used to propagate $\mathrm{Z}$ values.

### Sign

```MLIR
%sgn = mvl.sign %s : !mvl.logic<N>
```
is _defined_ to be equal to

```MLIR
%pre   = mvl.literal "0" : !mvl.logic<1>
%sgn   = mvl.insert %pre, %s, (1 - N) : !mvl.logic<1>, !mvl.logic<N>
```

### Identity
```mvl.identity``` returns the unchanged value of its operand. It is only useful in conjunction with an [Abstract Upper Limit Attribute (AULA)](#abstract-upper-limit-attribute) .

## Poison
Irrespective of width, any ```!mvl.logic``` typed SSA value can become $\mathrm{Poison}$. $\mathrm{Poison}$ is a special value that only exists during compilation and indicates the violation of an assumption (see ```ub.poison```). A $\mathrm{Poison}$ value can be _refined_ to an arbitrary non-poison value supported by the given type at any point during compilation. Unless explicitly stated otherwise, all MVL operations produce a $\mathrm{Poison}$ result, if any single of its operands is $\mathrm{Poison}$.

It is worth noting that for empty vectors the value $\mathrm{Poison}$ is distinct from $\varepsilon$: Consider the ```mvl.and_reduce``` operation, which maps a vector of any width to a vector of width one. The $\mathrm{AND}$ reduction of $\varepsilon$ is defined to be ' $\mathrm{1}$ ' . In contrast, the
$\mathrm{AND}$ reduction of $\mathrm{Poison}$ is $\mathrm{Poison}$ of width one, which in turn can legally be refined to _any_ value in $B$. Furthermore, note that while  $\mathrm{AND}(b, \mathrm{0}) = \mathrm{0}$ holds for all $b \in B$, $\mathrm{Poison}$  cannot be "masked out" by boolean logic operations:  $\mathrm{AND}(\mathrm{Poison}, \mathrm{0}) = \mathrm{Poison}$. Consequently, stating that the result of an operation is $\mathrm{Poison}$ is significantly different from stating that the result is _undefined_ or $\mathrm{X}$.

In summary:
Let $a$ be a value of type ```!mvl.logic``` and width $k \in \mathbb{Z}$, then:

```math
a \in {\{\mathrm{Poison}\}} \cup \begin{cases}
    B^k & \text{if } k > 0 \\ % & is your "\tab"-like command (it's a tab alignment character)
    \{\varepsilon\} & \text{otherwise}
\end{cases}
```

Thus, the possible set of values $V$ for any given ```!mvl.logic``` is:

```math
V = {\{\varepsilon, \mathrm{Poison}\}} \cup \bigcup_{n \in \mathbb{N^+}}B^n
```

### But why?

As stated above, $\mathrm{Poison}$ is used to indicate the violation of an assumption. In other words, a frontend can let an operation produce $\mathrm{Poison}$ under conditions which are assumed to never occur or be of no relevance. Hence, it is acceptable that $\mathrm{Poison}$ can allow "illogical" transformations like $AND(\mathrm{Poison}, \mathrm{0})\rightsquigarrow 1$. More specifically, a pure Verilog frontend can substitute all values that are unsupported by its 4-valued `logic` type with $\mathrm{Poison}$ and thus allow optimizations that would be illegal in VHDL (e.g., $\forall a \in \{\mathrm{0}, \mathrm{1}, \mathrm{X}, \mathrm{Z}\}:  XOR(a,\mathrm{X})=\mathrm{X}$, but $XOR(\mathrm{U},\mathrm{X})=\mathrm{U}$).

Also, recall the reduction example from above. Given the following MLIR snippet:
```MLIR
%a = mvl.literal ..."0" : !mvl.logic<#hw.param.decl.ref<"WIDTH">>
%b = mvl.and_reduce %a  : !mvl.logic<#hw.param.decl.ref<"WIDTH">>
```
An "obvious" Verilog lowering would be:
```Verilog
logic [WIDTH-1:0] a;
logic b;
assign a = '0; // mvl.literal
assign b = &a; // mvl.and_reduce
```
For $a \in \{0\}^{n} \cup \{\varepsilon\} \:,\: n \in \mathbb{N^+}$ the following equivalence holds: $\mathrm{ANDReduce}(a) = 1 \Leftrightarrow a = \varepsilon$, and for the given example, assuming $a \neq \mathrm{Poison}$: $a = \varepsilon \Leftrightarrow \mathrm{WIDTH} \leq 0$. \
However, Verilog does not support zero-width arrays. For ```WIDTH == 0``` the variable ``a`` would become a two-element array instead, and thus ```b == &(2'b00) == 1'b0```. Hence, the above lowering would have to be considered illegal and require additional checks on the ```WIDTH``` parameter. Allowing the Verilog frontend to substitute $\varepsilon$ results with $\mathrm{Poison}$ sidesteps this issue.

$\mathrm{Poison}$ can easily and efficiently be integrated into formal methods, such as translation validation. In contrast to LLVM's $\mathrm{undef}$, it avoids having to deal with sets of values. It also does not require existential quantification. [See the Alive2 paper for reference.](https://users.cs.utah.edu/~regehr/alive2-pldi21.pdf)

## Combinational Loops

Combinational loops in MVL are permitted but discouraged. They generally do not provide a useful model and can cause undesirable effects. Consider the following example:
```MLIR
%one = mvl.constant 1 : mvl.logic<1>
%a   = mvl.xor %a, %one :  mvl.logic<1>
```

In a typical HDL simulation, the behavior of ```%a``` will depend on its initial value. For initial values of $\mathrm{0}$ or $\mathrm{1}$ it will oscillate and likely prevent simulation progress. The MVL dialect explicitly does not model any state or time aspects. Consequently this behavior is beyond its semantics. \
Instead, the example can be interpreted as the algebraic equation $a = a \: XOR \: 1$ . With $a \in B \cup \{ \mathrm{Poison} \}$, this equation is solved by $a = \mathrm{X}$, $a = \mathrm{U}$ and $a = \mathrm{Poison}$. In fact, the solution of almost all pure MVL loops will include $\mathrm{Poison}$, effectively leading to _undefined behavior_. Thus, it is recommended that all circular dataflow paths pass through at least one operation that provides time or state semantics and does not propagate $\mathrm{Poison}$ values.

## Abstract Upper Limit Attribute

Every operation in the MVL dialect (except ```mvl.to_integer```) can carry an _Abstract Upper Limit Attribute_ (AULA), which can be used to restrict the "relevant" operation results to a subset of the values admitted by the result's type. Every result outside of this subset is replaced with $\mathrm{Poison}$. The AULA specifies this subset in the _abstract domain_ of permitted logic digits and $\varepsilon$.

Formally speaking, the AULA specifies a value within the bounded lattice $(L,\subseteq)$ defined over the power set of $B$ augmented with a "quasi-digit" $\mathrm{E}$ for the empty vector:
```math
L := \mathcal{P}(B \cup \{\mathrm{E}\}) \quad , \quad \bot = \emptyset \quad , \quad \top = B \cup \{\mathrm{E}\}
```

For any given MVL operation let $f:D \to V$ be the function providing the operation's semantics defined over its respective argument domain $D$. By specifing an abstract upper limit, we substitute the operation's result with a function $g$ that combines the original arguments $d \in D$ with the AULA $a \in L$ and an abstraction $\alpha$ mapping a set of concrete values to our abstract lattice:

```math
\alpha: \mathcal{P}(V) \to L  \quad , \quad g:D \times L \to V \quad, \quad g(d,a) := \begin{cases}
    f(d) & \text{if } \alpha(\{ f(d) \}) \subseteq a \\ % & is your "\tab"-like command (it's a tab alignment character)
    \mathrm{Poison} & \text{otherwise}
\end{cases}
```

The specific abstraction currently implemented by the MVL dialect is:

```math
\alpha (y) := \bigcup_{v \in y} \begin{cases}
    \bigcup_{0 \leq i < n } \{ v_{i} \} & \text{if } v \in B^n \:, \: n \in \mathbb{N^+}  \\ % & is your "\tab"-like command (it's a tab alignment character)
    \{\mathrm{E}\} & \text{if } v = \varepsilon \\
    \bot  & \text{if } v = \mathrm{Poison}
\end{cases}
```

Where $v_{i}$ denotes the element at index $i$ in the logic vector $v$. By mapping $\mathrm{Poison}$ to $\bot$ we ensure that the AULA indeed limits the operation's result in the abstract domain:
```math
\forall f,d,a: \alpha(\{ g(d,a) \} ) \subseteq a
```

We can also see that specifying no AULA is equivalent to using $\top$ as the limit:
```math
\forall f,d: g(d,\top) = f(d)
```

### Notation

If an MVL operation carries an AULA, it is denoted right after the operation name. An element of the lattitce $L$ can be specified in the explicit set notation. E.g., the following operation carries the AULA $\{ \mathrm{0} , \mathrm{1} , \mathrm{X}, \mathrm{Z}, \mathrm{-}, \mathrm{E}\} \in L$ :
```MLIR
%0 = mvl.identity [b0,b1,X,Z,DC,E] %a : !mvl.logic<N>
```

For frequently used combinations a shorthand notation is implemented:

* ```bin``` corresponds to  $\{ \mathrm{0} , \mathrm{1} \}$
* ```xb``` corresponds to  $\{ \mathrm{0} , \mathrm{1} , \mathrm{X} \}$
* ```vlog``` corresponds to  $\{ \mathrm{0} , \mathrm{1} , \mathrm{X} , \mathrm{Z} \}$

### Example

For a more practical perspective, let's revisit the example from above. Adding the "default" Verilog AULA $\{\mathrm{0},\mathrm{1},\mathrm{X},\mathrm{Z}\}$ produces the following IR dump:

```MLIR
%a = mvl.literal vlog ..."0" : !mvl.logic<#hw.param.decl.ref<"WIDTH">>
%b = mvl.and_reduce vlog %a  : !mvl.logic<#hw.param.decl.ref<"WIDTH">>
```

Obviously, the zero literal will never contain a $\mathrm{1}$, $\mathrm{X}$, or $\mathrm{Z}$ digit, and from the definition of ```mvl.and_reduce``` it follows that its result will never be $\mathrm{Z}$ for any input. So, just going by the individual operation's _abstract semantics_ we can easily simplify this to:

```MLIR
%a = mvl.literal [b0] ..."0" : !mvl.logic<#hw.param.decl.ref<"WIDTH">>
%b = mvl.and_reduce xb %a  : !mvl.logic<#hw.param.decl.ref<"WIDTH">>
```

Looking even closer at the abstract semantics of ```mvl.and_reduce```, we can see that its result can only be $\mathrm{1}$ or $\mathrm{X}$ if its input is $\varepsilon$ or contains any of the following digits: $\mathrm{1}$, $\mathrm{H}$, $\mathrm{W}$, $\mathrm{X}$, $\mathrm{Z}$, $\mathrm{-}$. Since all of these options are ruled out by the AULA of ```%a```'s defining operation, we can boil it down further to.

```MLIR
%a = mvl.literal [b0] ..."0" : !mvl.logic<#hw.param.decl.ref<"WIDTH">>
%b = mvl.and_reduce [b0] %a  : !mvl.logic<#hw.param.decl.ref<"WIDTH">>
```

Considering that the result type of a reduction operation is ```!mvl.logic<1>```, the only possible concrete values left for ```%b``` are ' $\mathrm{0}$ ' and $\mathrm{Poison}$. Since we are free to refine $\mathrm{Poison}$ to any value we like, we make the obvious choice:

```MLIR
%a = mvl.literal [b0] ..."0" : !mvl.logic<#hw.param.decl.ref<"WIDTH">>
%b = mvl.constant [b0] 0 : !mvl.logic<1>
```

## Verilog Idioms

### Case Equality ( === )

```Verilog
a === b
```

```MLIR
%elteq  = mvl.eltwise_eq bin %a, %b : !mvl.logic<N>
%caseeq = mvl.and_reduce bin %elteq : !mvl.logic<N>
```

### Logical Equality ( == )

```Verilog
a == b
```

```MLIR
%equals = mvl.match xb %a, %b : !mvl.logic<N>
```

### Wildcard Equality ( ==? )

```Verilog
a ==? b
```

```MLIR
%allx = mvl.literal ..."X" : !mvl.logic<N>
%conv = mvl.to_x01 xb %b : : !mvl.logic<N>
%mask = mvl.eltwise_eq bin %conv, %allx : !mvl.logic<N>
%wceq = mvl.match xb %a, %b, %mask : !mvl.logic<N>
```

### Trinary operator
```Verilog
s ? a : b
```

```MLIR
%x    = mvl.literal "X" : !mvl.logic<1>
%conv = mvl.to_x01 xb %s : : !mvl.logic<1>
%isxz = mvl.eltwise_eq bin %conv, %x : !mvl.logic<1>
%xmux = mvl.mux xb %s, %a, %b : !mvl.logic<N>
%sel  = mvl.select vlog %s, %a, %b : !mvl.logic<N>
%res  = mvl.select vlog %isxz, %xmux, %sel : !mvl.logic<N>
```

### 4-State to 2-State Cast
```Verilog
logic [N-1:0] a;
bit   [N-1:0] b;
...
assign b = a;
```

```MLIR
%ones = mvl.literal ..."1" : !mvl.logic<N>
%b  = mvl.eltwise_eq bin %a, %ones : !mvl.logic<N>
```

### If Else
```Verilog
if (cond)
  res = a;
else
  res = b;
```

```MLIR
%one  = mvl.literal "1" : !mvl.logic<1>
%rcon = mvl.or_reduce vlog %cond : !mvl.logic<N>
%scon = mvl.eltwise_eq %rcon, %one : !mvl.logic<1>
%res  = mvl.select vlog %scon, %a, %b  : !mvl.logic<M>
```

### Logical / Unsigned Artihmetic Shift Right ( >> / >>> )

```Verilog
shiftee >> shiftamt
$unsigned(shiftee) >>> shiftamt
```

```MLIR
%zeros = mvl.literal ..."0" : !mvl.logic<N>
%x = mvl.literal ..."X" : !mvl.logic<N>
%shr = mvl.int_or_else vlog (%shamt : !mvl.logic<M>) else %x : !mvl.logic<N> {
  ^bb0(%intarg0: !hw.int<M>):
  %0 = mvl.dyn_insert %zeros, %shiftee, NEG %intarg0 : !mvl.logic<N>, !mvl.logic<N>, !hw.int<M>
  mvl.yield %0 : !mvl.logic<N>
}
```

### Signed Artihmetic Shift Right ( >>> )

```Verilog
$signed(shiftee) >>> shiftamt
```

```MLIR
%x = mvl.literal ..."X" : !mvl.logic<N>
%sign  = mvl.sign %shiftee : !mvl.logic<N>
%fill  = mvl.sext_or_trunc %sign, N : !mvl.logic<N>
%shr = mvl.int_or_else vlog (%shamt : !mvl.logic<M>) else %x : !mvl.logic<N> {
  ^bb0(%intarg0: !hw.int<M>):
  %0 = mvl.dyn_insert %fill, %shiftee, NEG %intarg0 : !mvl.logic<N>, !mvl.logic<N>, !hw.int<M>
  mvl.yield %0 : !mvl.logic<N>
}
```

### Logical / Arithmetic Shift Left ( << / <<< )

```Verilog
shiftee << shiftamt
shiftee <<< shiftamt
```

```MLIR
%zeros = mvl.literal ..."0" : !mvl.logic<N>
%x = mvl.literal ..."X" : !mvl.logic<N>
%shl = mvl.int_or_else vlog (%shamt : !mvl.logic<M>) else %x : !mvl.logic<N> {
  ^bb0(%intarg0: !hw.int<M>):
  %0 = mvl.dyn_insert %zeros, %shiftee, POS %intarg0 : !mvl.logic<N>, !mvl.logic<N>, !hw.int<M>
  mvl.yield %0 : !mvl.logic<N>
}
```

### Concatenation

```Verilog
logic [K-1:0] a;
logic [N-1:0] b;
logic [M-1:0] c;
logic [K+M+N-1:0] cat;

assign cat = {c, b, a};
```


```MLIR
%zeros = mvl.literal ..."0" : !mvl.logic<K+M+N>
%tm0 = mvl.insert vlog %zeros, %a, 0 : !mvl.logic<K+M+N>, !mvl.logic<K>
%tm1 = mvl.insert vlog %0, %b, K : !mvl.logic<K+M+N>, !mvl.logic<N>
%cat = mvl.insert vlog %1, %c, (N + K): !mvl.logic<K+M+N>, !mvl.logic<M>
```