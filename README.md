# Inn Expression Evaluator

A lightweight, easy-to-use boolean expression evaluator designed for embedding in configuration files and other settings that need simple conditional logic.

## Features

- JIT (Just-In-Time) execution for dynamic evaluation
- AOT (Ahead-of-Time) compilation for optimized repeated execution (Work in Progress)
- Support for boolean, integer and string values
- Variables and string interpolation
- Prepared statements with parameter binding
- Minimal dependencies

## Usage

### JIT Mode (Default)
```rust
let mut vm = VM::new();
vm.set_value("enabled", Value::Bool(true));
vm.set_value("count", Value::Int(5));

// Evaluate expression
let result = vm.exec("enabled and count > 3").unwrap();
```

### AOT Mode
```rust
let mut vm = VM::new();
vm.set_value("x", Value::Int(10));

// Compile once
let exe = vm.compile("x > 5 and x < 20").unwrap();

// Execute multiple times efficiently
let result = vm.exec_aot(&exe).unwrap();
```

#### VMExec

Is a format to store AOT instructions, not limit to any format serde-compatible. Why not store AST for JIT? I prefer so, actually less overhead on what being executed, but it actually more?. Updates:

- `0x2`:
    - Added `ConstPtr` and field `consts`


## Supported Operations

- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`, `in`
- Boolean: `and`, `or`, `xor`, `not`
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `^`
- String interpolation: `"Value: ${x}"`

## Data types

- Int: 32bit int
- Bool: 8bit bool
- String: Heap string
- Imaginary: Nil (is not derive Int)
- Abyss: Null

## Use Cases

- Configuration file conditions
- Simple decision trees
- Dynamic content filtering

## AOT Mode Implementation

Inn uses a B-tree (balanced tree) structure for AOT compilation to optimize instruction execution:

- Root node contains the final operation
- Inner nodes represent sub-operations
- Leaf nodes contain values (constants, heap refs, stack refs)
- Register allocation tracks usage across branches
- Tree height determines maximum register usage (limited to 8)

Example compilation:
```rust
// Expression: '(a and b) or (c and d)'
// Parsed into AST (generated through `parse()`):
let expr = Exp::Op(
    Box::new(Exp::Op(
        Box::new(Exp::Val("a".to_string())),
        Op::And,
        Box::new(Exp::Val("b".to_string()))
    )),
    Op::Or,
    Box::new(Exp::Op(
        Box::new(Exp::Val("c".to_string())),
        Op::And,
        Box::new(Exp::Val("d".to_string()))
    ))
);

let ops = vm.compile(expr).unwrap();

/* Expression generates B-tree:
        OR(r0)
       /      \
   AND(r0)  AND(r1)
   /    \    /    \
 a(h0) b(h1) c(h2) d(h3)

 Generates AOT operations:
 1. r0 = HeapPTR("a") AND HeapPTR("b")
 2. r1 = HeapPTR("c") AND HeapPTR("d")
 3. r0 = r0 OR r1
*/
```

The balanced tree structure:
- Ensures optimal register usage
- Minimizes instruction count
- Enables parallel evaluation
- Supports partial evaluation optimization

## State of the optimization

Whatever the chart is, in many non complex case, JIT beats off AOT with itself plus the AST (same thing AOT use)

```
        min          mean           max
AOT
time:   [4.1950 µs 4.2042 µs 4.2138 µs]
JIT
time:   [3.1339 µs 3.1400 µs 3.1460 µs]
```

### talktuah

Current state, AOT is kinda slow, for some reason. The main reason I think is it have many `clone` overhead. 
Meanwhile doing for new datatype, I accidentally added 10% overhead for JIT, oopsie
