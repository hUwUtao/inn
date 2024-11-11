//! A simple virtual machine for logical and math expressions.
use std::collections::HashMap;

/// Supported operators for the VM
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Op {
    Is,          // ==
    Not,         // !=
    And,         // &&
    Or,          // ||
    Xor,         // ^
    UpTo,        // <=
    DownTo,      // >=
    GreaterThan, // >
    LesserThan,  // <
    In,          // contains
    Add,         // +
    Subtract,    // -
    Multiply,    // *
    Divide,      // /
    Power,       // ^
    Modulo,      // %
    Fmt,         // String format
}

/// Represents the types of values supported by the VM
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Value {
    /// Boolean true/false value
    Bool(bool),
    /// 64-bit signed integer value
    Int(i32),
    /// String value
    String(String),
    /// Null/absent value
    Abyss,
    /// Math error
    Imaginary,
}

/// Represents an expression in the abstract syntax tree (AST)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Exp {
    /// Variable or literal value node
    Val(String),
    /// Binary operation node with left operand, operator, and right operand
    Op(Box<Exp>, Op, Box<Exp>),
}

/// The value defined in compiled instruction
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VMExp {
    /// Reference to a variable in the global heap
    HeapVar(String),
    /// Reference to a variable in the local heap
    PHeapPtr(u8),
    /// Reference to a value on the stack
    StackPtr(u8),
    /// Reference to a value in program
    ConstPtr(u16),
    /// A literal value
    Literal(Value),
}

/// Represents a VM operation
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VMOp {
    /// Destination register number (0-7)
    pub reg: u8,
    /// Operation to perform
    pub op: Op,
    /// Left operand expression
    pub l: VMExp,
    /// Right operand expression
    pub r: VMExp,
}

/// Represents an executable VM program
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VMExec {
    /// Version number of the VM format
    pub ver: u8,
    /// Required private heap size in bytes
    pub private_heap: u16,
    /// Shared const
    pub consts: Vec<Value>,
    /// List of instructions (operation) to execute
    pub inst: Vec<VMOp>,
}

/// Register tracker for managing registers during compilation
struct RegisterTracker {
    tracker: [bool; 8],
}

/// The virtual machine that handles parsing and evaluation
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VM {
    heap_table: HashMap<String, Value>,
}

// Original implementations follow...
impl RegisterTracker {
    fn new() -> Self {
        Self {
            tracker: [false; 8],
        }
    }

    fn allocate(&mut self) -> u8 {
        if let Some(reg) = (0..8).find(|&r| !self.tracker[r as usize]) {
            self.tracker[reg as usize] = true;
            reg
        } else {
            panic!("Register overflow - no more registers available. Compiler exceed overhead.");
        }
    }

    fn release(&mut self, reg: u8) {
        if reg >= 8 {
            panic!("Invalid register number");
        }
        self.tracker[reg as usize] = false;
    }
}

impl VMExec {
    pub fn compile(ast: Exp) -> Result<Self, String> {
        let mut ops = Vec::new();
        let mut reg_tracker = RegisterTracker::new();
        let mut consts = Vec::new();
        let mut const_map = std::collections::HashMap::new();

        let _ = Self::compile_ast(
            &mut ops,
            &ast,
            &mut reg_tracker,
            &mut consts,
            &mut const_map,
        )?;

        Ok(VMExec {
            ver: 0x02,
            private_heap: 256,
            consts,
            inst: ops,
        })
    }

    fn compile_ast(
        ops: &mut Vec<VMOp>,
        ast: &Exp,
        reg: &mut RegisterTracker,
        consts: &mut Vec<Value>,
        const_map: &mut std::collections::HashMap<Value, u16>,
    ) -> Result<VMExp, String> {
        match ast {
            Exp::Val(v) => {
                let value = if v.starts_with('\'') || v.starts_with('"') {
                    let content = &v[1..v.len() - 1];
                    if content.contains('{') && content.contains('}') {
                        // Break the string into parts and create fmt operations
                        let mut parts: Vec<Box<Exp>> = Vec::new();
                        let mut current = String::new();
                        let mut in_expr = false;
                        let mut expr_start = 0;

                        let chars: Vec<_> = content.chars().collect();
                        let mut i = 0;
                        while i < chars.len() {
                            if chars[i] == '{' && i + 1 < chars.len() && chars[i + 1] == '}' {
                                if !current.is_empty() {
                                    parts.push(Box::new(Exp::Val(
                                        String::from("'") + &current + "'",
                                    )));
                                    current.clear();
                                }
                                in_expr = true;
                                expr_start = i;
                                i += 2;
                            } else {
                                if in_expr {
                                    in_expr = false;
                                    let expr_str = &content[expr_start + 1..i - 1];
                                    let expr = VM::parse(expr_str)?;
                                    parts.push(Box::new(expr));
                                }
                                current.push(chars[i]);
                                i += 1;
                            }
                        }

                        if !current.is_empty() {
                            let mut quoted = String::with_capacity(current.len() + 2);
                            quoted.push('\'');
                            quoted.push_str(&current);
                            quoted.push('\'');
                            parts.push(Exp::val(quoted));
                        }

                        // Chain parts with fmt operators
                        let mut result = parts[0].clone();
                        for part in parts.iter().skip(1) {
                            result = Exp::op(result, Op::Fmt, part.clone());
                        }

                        return Self::compile_ast(ops, &result, reg, consts, const_map);
                    } else {
                        Value::String(content.to_string())
                    }
                } else if let Ok(num) = v.parse::<i32>() {
                    Value::Int(num)
                } else {
                    match v.to_lowercase().as_str() {
                        "true" => Value::Bool(true),
                        "false" => Value::Bool(false),
                        _ => return Ok(VMExp::HeapVar(v.clone())),
                    }
                };

                // Deduplicate constants
                if let Some(idx) = const_map.get(&value) {
                    Ok(VMExp::ConstPtr(*idx))
                } else {
                    let idx = consts.len() as u16;
                    const_map.insert(value.clone(), idx);
                    consts.push(value);
                    Ok(VMExp::ConstPtr(idx))
                }
            }
            Exp::Op(left, op, right) => {
                // Compile sub-expressions
                let left_exp = Self::compile_ast(ops, left, reg, consts, const_map)?;
                let right_exp = Self::compile_ast(ops, right, reg, consts, const_map)?;
                // Allocate register for result
                let result_reg = reg.allocate();
                // Release values register
                if let VMExp::StackPtr(ptr) = left_exp {
                    reg.release(ptr);
                }
                if let VMExp::StackPtr(ptr) = right_exp {
                    reg.release(ptr);
                }
                // Add operation
                ops.push(VMOp {
                    reg: result_reg,
                    op: *op,
                    l: left_exp,
                    r: right_exp,
                });

                Ok(VMExp::StackPtr(result_reg))
            }
        }
    }
}

impl Value {
    #[inline]
    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::String(s) => !s.is_empty(),
            Value::Abyss => false,
            Value::Imaginary => false,
        }
    }

    #[inline]
    fn as_int(&self) -> i32 {
        match self {
            Value::Int(i) => *i,
            Value::Bool(b) => *b as i32,
            Value::String(s) => s.parse::<i32>().unwrap_or(0),
            Value::Abyss => 0,
            // Last resort
            Value::Imaginary => i32::MIN,
        }
    }

    #[inline]
    fn as_string(&self) -> String {
        match self {
            Value::String(s) => s.clone(),
            Value::Int(i) => i.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Abyss => String::from("::abyss"),
            Value::Imaginary => String::from("::i"),
        }
    }
}

impl Exp {
    /// Creates a new operator expression node
    #[inline]
    pub fn op(left: Box<Exp>, op: Op, right: Box<Exp>) -> Box<Exp> {
        Box::new(Exp::Op(left, op, right))
    }

    /// Creates a new value expression node
    #[inline]
    pub fn val(name: String) -> Box<Exp> {
        Box::new(Exp::Val(name))
    }
}

impl VM {
    /// Creates a new VM instance with an empty global context
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a value in the global context
    #[inline]
    pub fn set_value(&mut self, name: &str, value: Value) {
        self.heap_table.insert(name.to_string(), value);
    }

    /// Gets a value from the global context
    #[inline]
    pub fn get_value(&self, name: &str) -> Option<&Value> {
        self.heap_table.get(name)
    }

    /// Execute an expression string
    #[inline]
    pub fn exec(&self, expr: &str) -> Result<Value, String> {
        self.eval(&VM::parse(expr)?)
    }

    /// Execute an expression string with mutable VM access
    #[inline]
    pub fn exec_mut(&mut self, expr: &str) -> Result<Value, String> {
        self.eval(&VM::parse(expr)?)
    }

    /// Evaluates an expression (in JIT)
    pub fn eval(&self, expr: &Exp) -> Result<Value, String> {
        match expr {
            Exp::Val(name) => {
                if name.starts_with('\'') || name.starts_with('"') {
                    let content = &name[1..name.len() - 1];
                    Ok(Value::String(content.to_string()))
                } else if let Ok(num) = name.parse::<i32>() {
                    Ok(Value::Int(num))
                } else if let Some(val) = self.heap_table.get(name) {
                    Ok(val.clone())
                } else {
                    match name.to_lowercase().as_str() {
                        "true" => Ok(Value::Bool(true)),
                        "false" => Ok(Value::Bool(false)),
                        _ => Ok(Value::Abyss),
                    }
                }
            }
            Exp::Op(left, op, right) => {
                let l = self.eval(left)?;
                let r = self.eval(right)?;
                match (op, &l, &r) {
                    // Handle math operations with Imaginary and Abyss
                    (
                        Op::Add | Op::Subtract | Op::Multiply | Op::Divide | Op::Power | Op::Modulo,
                        Value::Imaginary,
                        _,
                    )
                    | (
                        Op::Add | Op::Subtract | Op::Multiply | Op::Divide | Op::Power | Op::Modulo,
                        _,
                        Value::Imaginary,
                    ) => Ok(Value::Imaginary),
                    _ => VM::eval_operation(op, l, r),
                }
            }
        }
    }

    /// Evaluate an executable
    #[inline]
    pub fn exec_aot(&self, exec: &VMExec) -> Result<Box<Value>, String> {
        let mut stack = vec![None; 8];

        #[inline]
        fn get_operand<'a>(
            exp: &'a VMExp,
            stack: &'a Vec<Option<Value>>,
            heap: &'a HashMap<String, Value>,
            consts: &'a [Value],
        ) -> Result<Value, String> {
            match exp {
                VMExp::HeapVar(ref name) => Ok(heap.get(name).cloned().unwrap_or(Value::Abyss)),
                VMExp::PHeapPtr(_) => todo!("Private Heap is not implemented"),
                VMExp::StackPtr(idx) => stack[*idx as usize]
                    .clone()
                    .ok_or_else(|| format!("Value not found at stack index {}", idx)),
                VMExp::ConstPtr(idx) => Ok(consts[*idx as usize].clone()),
                VMExp::Literal(ref v) => Ok(v.clone()),
            }
        }

        for inst in &exec.inst {
            let l = get_operand(&inst.l, &stack, &self.heap_table, &exec.consts)?;
            let r = get_operand(&inst.r, &stack, &self.heap_table, &exec.consts)?;

            let result = match (&inst.op, &l, &r) {
                // Handle math operations
                (
                    Op::Add | Op::Subtract | Op::Multiply | Op::Divide | Op::Power | Op::Modulo,
                    Value::Imaginary,
                    _,
                )
                | (
                    Op::Add | Op::Subtract | Op::Multiply | Op::Divide | Op::Power | Op::Modulo,
                    _,
                    Value::Imaginary,
                ) => Ok(Value::Imaginary),
                _ => VM::eval_operation(&inst.op, l, r),
            }?;

            stack[inst.reg as usize] = Some(result);
        }

        if let Some(last_inst) = exec.inst.last() {
            stack[last_inst.reg as usize]
                .clone()
                .map(Box::new)
                .ok_or_else(|| "No result".to_string())
        } else {
            Err("No instructions".to_string())
        }
    }

    fn parse_expr(tokens: &[String]) -> Result<Exp, String> {
        let mut pos = 0;
        let capacity = tokens.len() / 2;
        let mut stack = Vec::with_capacity(capacity);
        let mut operators = Vec::with_capacity(capacity);

        while pos < tokens.len() {
            let token = &tokens[pos];
            if token.starts_with('\'') || token.starts_with('"') {
                let content = &token[1..token.len() - 1];
                if content.contains('{') && content.contains('}') {
                    let mut parts = Vec::new();
                    let mut current = String::new();
                    let mut i = 0;
                    let chars: Vec<_> = content.chars().collect();

                    // Split into parts and expressions
                    while i < chars.len() {
                        if chars[i] == '{' {
                            if !current.is_empty() {
                                parts.push(Exp::Val(format!("'{}'", current)));
                                current.clear();
                            }
                            i += 1; // Skip {
                            let mut expr = String::new();
                            while i < chars.len() && chars[i] != '}' {
                                expr.push(chars[i]);
                                i += 1;
                            }
                            if i < chars.len() {
                                // Parse the expression inside {}
                                let parsed = VM::parse(&expr)?;
                                parts.push(parsed);
                            }
                            i += 1; // Skip }
                        } else {
                            current.push(chars[i]);
                            i += 1;
                        }
                    }

                    if !current.is_empty() {
                        parts.push(Exp::Val(format!("'{}'", current)));
                    }

                    // Chain parts with fmt operators
                    let mut expr = parts[0].clone();
                    for part in parts.iter().skip(1) {
                        expr = Exp::Op(Box::new(expr), Op::Fmt, Box::new(part.clone()));
                    }
                    stack.push(expr);
                } else {
                    stack.push(Exp::Val(token.clone()));
                }
            } else {
                match &token[..] {
                    "(" => {
                        operators.push(token.clone());
                    }
                    ")" => {
                        while let Some(op) = operators.last() {
                            if op == "(" {
                                operators.pop();
                                break;
                            }
                            VM::apply_operator(&mut stack, operators.pop().unwrap())?;
                        }
                    }
                    "+" | "-" | "*" | "/" | "^" | "%" | "and" | "or" | "xor" | "not" | "is"
                    | "<=" | ">=" | ">" | "<" | "in" => {
                        while let Some(op) = operators.last() {
                            if op == "(" || VM::precedence(op) < VM::precedence(token) {
                                break;
                            }
                            VM::apply_operator(&mut stack, operators.pop().unwrap())?;
                        }
                        operators.push(token.clone());
                    }
                    _ => {
                        stack.push(Exp::Val(token.clone()));
                    }
                }
            }
            pos += 1;
        }

        while let Some(op) = operators.pop() {
            if op == "(" {
                return Err("Unclosed parenthesis".to_string());
            }
            VM::apply_operator(&mut stack, op)?;
        }

        if stack.len() == 1 {
            Ok(stack.pop().unwrap())
        } else {
            Err("Invalid expression".to_string())
        }
    }

    fn eval_operation(op: &Op, left: Value, right: Value) -> Result<Value, String> {
        match op {
            // Math operations - convert bools to ints first
            Op::Add => Ok(Value::Int(left.as_int() + right.as_int())),
            Op::Subtract => Ok(Value::Int(left.as_int() - right.as_int())),
            Op::Multiply => Ok(Value::Int(left.as_int() * right.as_int())),
            Op::Divide => {
                let r = right.as_int();
                if r == 0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(Value::Int(left.as_int() / r))
                }
            }
            Op::Power => {
                let r = right.as_int();
                if r < 0 {
                    Err("Negative exponent".to_string())
                } else {
                    Ok(Value::Int(left.as_int().pow(r as u32)))
                }
            }
            Op::Modulo => {
                let r = right.as_int();
                if r == 0 {
                    Err("Modulo by zero".to_string())
                } else {
                    Ok(Value::Int(left.as_int() % r))
                }
            }

            // Boolean operations
            Op::Is => Ok(Value::Bool(left.as_int() == right.as_int())),
            Op::Not => Ok(Value::Bool(left.as_int() != right.as_int())),
            Op::And => Ok(Value::Bool(left.as_bool() && right.as_bool())),
            Op::Or => Ok(Value::Bool(left.as_bool() || right.as_bool())),
            Op::Xor => Ok(Value::Bool(left.as_bool() ^ right.as_bool())),
            Op::UpTo => Ok(Value::Bool(left.as_int() <= right.as_int())),
            Op::DownTo => Ok(Value::Bool(left.as_int() >= right.as_int())),
            Op::GreaterThan => Ok(Value::Bool(left.as_int() > right.as_int())),
            Op::LesserThan => Ok(Value::Bool(left.as_int() < right.as_int())),
            Op::In => match (&left, &right) {
                (Value::String(needle), Value::String(haystack)) => {
                    Ok(Value::Bool(haystack.contains(needle)))
                }
                (Value::Int(num), Value::String(s)) => {
                    Ok(Value::Bool(s.contains(&num.to_string())))
                }
                (Value::String(s), Value::Int(num)) => {
                    Ok(Value::Bool(s.contains(&num.to_string())))
                }
                _ => Err("Invalid operation for types".to_string()),
            },

            // String formatting
            Op::Fmt => {
                let mut result = left.as_string();
                result.push_str(&right.as_string());
                Ok(Value::String(result))
            }
        }
    }

    /// Parses a string into an expression
    pub fn parse(input: &str) -> Result<Exp, String> {
        let tokens = VM::tokenize(input)?;
        VM::parse_expr(&tokens)
    }

    fn tokenize(input: &str) -> Result<Vec<String>, String> {
        let mut tokens = Vec::with_capacity(input.len() / 2);
        let mut chars = input.chars().peekable();
        let mut current = String::with_capacity(32);

        while let Some(c) = chars.next() {
            match c {
                '(' | ')' | '<' | '>' | '=' | '+' | '-' | '*' | '/' | '^' | '%' => {
                    if !current.is_empty() {
                        tokens.push(current);
                        current = String::with_capacity(32);
                    }

                    // Handle two-character operators
                    if let Some(&next) = chars.peek() {
                        if (c == '<' && next == '=') || (c == '>' && next == '=') {
                            chars.next();
                            let mut op = String::with_capacity(2);
                            op.push(c);
                            op.push(next);
                            tokens.push(op);
                            continue;
                        }
                    }

                    tokens.push(c.to_string());
                }
                ' ' => {
                    if !current.is_empty() {
                        tokens.push(current);
                        current = String::with_capacity(32);
                    }
                }
                '\'' | '"' => {
                    if !current.is_empty() {
                        tokens.push(current);
                        current = String::with_capacity(32);
                    }
                    let quote = c;
                    current.push(quote);
                    while let Some(c) = chars.next() {
                        current.push(c);
                        if c == quote {
                            break;
                        }
                    }
                    tokens.push(current);
                    current = String::with_capacity(32);
                }
                _ => current.push(c),
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        Ok(tokens)
    }

    #[inline]
    fn precedence(op: &str) -> i32 {
        match op {
            "(" | ")" => 0,
            "or" => 1,
            "xor" => 2,
            "and" => 3,
            "not" | "is" => 4,
            "<=" | ">=" | ">" | "<" | "in" => 5,
            "+" | "-" => 6,
            "*" | "/" | "%" => 7,
            "^" => 8,
            _ => 0,
        }
    }

    #[inline]
    fn apply_operator(stack: &mut Vec<Exp>, op: String) -> Result<(), String> {
        if stack.len() < 2 {
            return Err("Invalid expression".to_string());
        }
        let right = stack.pop().unwrap();
        let left = stack.pop().unwrap();
        let operation = match &op[..] {
            "+" => Op::Add,
            "-" => Op::Subtract,
            "*" => Op::Multiply,
            "/" => Op::Divide,
            "^" => Op::Power,
            "%" => Op::Modulo,
            "and" => Op::And,
            "or" => Op::Or,
            "xor" => Op::Xor,
            "not" => Op::Not,
            "is" => Op::Is,
            "<=" => Op::UpTo,
            ">=" => Op::DownTo,
            ">" => Op::GreaterThan,
            "<" => Op::LesserThan,
            "in" => Op::In,
            _ => return Err(format!("Unknown operator: {}", op)),
        };
        stack.push(Exp::Op(Box::new(left), operation, Box::new(right)));
        Ok(())
    }
}
