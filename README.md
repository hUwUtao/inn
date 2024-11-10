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

### AOT Mode (WIP)
```rust
let mut vm = VM::new();
vm.set_value("x", Value::Int(10));

// Compile once
let ops = vm.compile("x > 5 and x < 20").unwrap();

// Execute multiple times efficiently
let result = vm.eval_aot(&ops).unwrap();
```

## Supported Operations

- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`, `in`
- Boolean: `and`, `or`, `xor`, `not`
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `^`
- String interpolation: `"Value: ${x}"`

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
// Expression: (a and b) or (c and d)
// Expression: '(a and b) or (c and d)'
// Parsed into AST:
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

/* Generates B-tree:
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
*/
```

The balanced tree structure:
- Ensures optimal register usage
- Minimizes instruction count
- Enables parallel evaluation
- Supports partial evaluation optimization
