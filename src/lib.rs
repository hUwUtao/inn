//! A simple virtual machine for logical and math expressions.
use std::collections::HashMap;
/// Represents the types of values supported by the VM
///
/// Includes boolean, integer and string values that can be stored and manipulated
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Value {
    /// Boolean true/false value
    Bool(bool),
    /// 64-bit signed integer value
    Int(i32),
    /// String value
    String(String),
}
/// Represents an executable VM program
///
/// Contains version info, memory requirements and compiled operations
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VMExec {
    /// Version number of the VM format
    pub ver: u8,
    /// Required private heap size in bytes
    pub private_heap: u16,
    /// List of operations to execute
    pub ops: Vec<VMOp>,
}

/// Register tracker
///
/// Once the tree expanded, manage heap will be a big ol problem.
struct RegisterTracker {
    tracker: [bool; 8],
}

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

        let _ = Self::compile_ast(&mut ops, &ast, &mut reg_tracker)?;

        Ok(VMExec {
            ver: 1,
            private_heap: 256,
            ops,
        })
    }

    fn parse_val(s: &str) -> Result<Value, String> {
        if s.starts_with('\'') || s.starts_with('"') {
            let content = &s[1..s.len() - 1];
            Ok(Value::String(content.to_string()))
        } else if let Ok(num) = s.parse::<i32>() {
            Ok(Value::Int(num))
        } else {
            match s.to_lowercase().as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => Err(format!("Invalid value: {}", s)),
            }
        }
    }

    fn compile_ast(
        ops: &mut Vec<VMOp>,
        ast: &Exp,
        reg: &mut RegisterTracker,
    ) -> Result<VMExp, String> {
        match ast {
            Exp::Val(v) => {
                let value = if let Ok(val) = Self::parse_val(v) {
                    VMExp::Literal(val)
                } else {
                    VMExp::HeapVar(v.clone())
                };
                Ok(value)
            }
            Exp::Op(left, op, right) => {
                // Compile sub-expressions
                let left_exp = Self::compile_ast(ops, left, reg)?;
                let right_exp = Self::compile_ast(ops, right, reg)?;
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

/// Represents a VM operation
///
/// Contains the destination register, operator, left operand, and right operand for executing instructions
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

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// The value defined in compiled instruction
pub enum VMExp {
    /// Reference to a variable in the global heap
    HeapVar(String),
    /// Reference to a variable in the local heap
    PHeapPtr(u8),
    /// Reference to a value on the stack
    StackPtr(u8),
    /// A literal value
    Literal(Value),
    // /// A string literal
    // String(String),
}

impl Value {
    #[inline]
    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::String(s) => !s.is_empty(),
        }
    }

    #[inline]
    fn as_int(&self) -> i32 {
        match self {
            Value::Int(i) => *i,
            Value::Bool(b) => *b as i32,
            Value::String(s) => s.parse::<i32>().unwrap_or(0),
        }
    }
}

/// Represents a prepared statement with parameters
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Stmt {
    expr: String,
    params: Vec<Value>,
}

impl Stmt {
    /// Binds a value to the next ? placeholder
    pub fn bind(mut self, value: Value) -> Self {
        self.params.push(value);
        self
    }
}

/// Represents an expression in the abstract syntax tree (AST)
///
/// Can be either a value (variable/literal) or an operation between expressions
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Exp {
    /// Variable or literal value node
    Val(String),
    /// Binary operation node with left operand, operator, and right operand
    Op(Box<Exp>, Op, Box<Exp>),
}

impl Exp {
    /// Creates a new operator expression node
    ///
    /// # Arguments
    /// * `left` - Left operand expression
    /// * `op` - Operator
    /// * `right` - Right operand expression
    #[inline]
    pub fn op(left: Box<Exp>, op: Op, right: Box<Exp>) -> Box<Exp> {
        Box::new(Exp::Op(left, op, right))
    }

    /// Creates a new value expression node
    ///
    /// # Arguments
    /// * `name` - String value or variable name
    #[inline]
    pub fn val(name: String) -> Box<Exp> {
        Box::new(Exp::Val(name))
    }
}

/// Supported operators
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
}

/// The virtual machine that handles parsing and evaluation of boolean expressions
///
/// Maintains a heap table mapping variable names to values and provides
/// methods for parsing and evaluating expressions.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VM {
    heap_table: HashMap<String, Value>,
}

impl VM {
    /// Creates a new VM instance with an empty global context
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new prepared statement
    #[inline]
    pub fn stmt(&self, expr: &str) -> Stmt {
        Stmt {
            expr: expr.to_string(),
            params: Vec::new(),
        }
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
        self.eval(&self.parse(expr)?)
    }

    /// Execute an expression string with mutable VM access
    #[inline]
    pub fn exec_mut(&mut self, expr: &str) -> Result<Value, String> {
        self.eval(&self.parse(expr)?)
    }

    /// Evaluates an expression
    #[inline]
    pub fn eval(&self, expr: &Exp) -> Result<Value, String> {
        match expr {
            Exp::Val(name) => self.eval_value(name),
            Exp::Op(left, op, right) => {
                let l = self.eval(left)?;
                let r = self.eval(right)?;
                self.eval_operation(op, l, r)
            }
        }
    }

    /// Executes a prepared statement with bound parameters
    pub fn exec_stmt(&self, stmt: Stmt) -> Result<Value, String> {
        let mut param_index = 0;
        let mut final_expr = String::with_capacity(stmt.expr.len());
        let mut in_string = false;
        let mut string_char = ' ';

        for c in stmt.expr.chars() {
            if !in_string && c == '?' {
                if param_index >= stmt.params.len() {
                    return Err("Not enough parameters".to_string());
                }
                let param = &stmt.params[param_index];
                let param_str = match param {
                    Value::Bool(b) => b.to_string(),
                    Value::Int(i) => i.to_string(),
                    Value::String(s) => format!("'{}'", s.replace('\'', "\\'")),
                };
                final_expr.push_str(&param_str);
                param_index += 1;
            } else {
                if c == '\'' || c == '"' {
                    if !in_string {
                        in_string = true;
                        string_char = c;
                    } else if c == string_char {
                        in_string = false;
                    }
                }
                final_expr.push(c);
            }
        }

        if param_index < stmt.params.len() {
            return Err("Too many parameters".to_string());
        }

        let expr = self.parse(&final_expr)?;
        self.eval(&expr)
    }

    #[inline]
    fn eval_value(&self, name: &str) -> Result<Value, String> {
        let value = self.parse_value(name)?;
        Ok(value)
    }

    fn eval_operation(&self, op: &Op, left: Value, right: Value) -> Result<Value, String> {
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
        }
    }

    /// Parses a string into an expression
    pub fn parse(&self, input: &str) -> Result<Exp, String> {
        let tokens = self.tokenize(input)?;
        self.parse_expr(&tokens)
    }

    fn tokenize(&self, input: &str) -> Result<Vec<String>, String> {
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
                            tokens.push(format!("{}{}", c, next));
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

    fn parse_value(&self, s: &str) -> Result<Value, String> {
        if s.starts_with('\'') || s.starts_with('"') {
            let content = &s[1..s.len() - 1];
            Ok(Value::String(self.interpolate_string(content)?))
        } else if let Ok(num) = s.parse::<i32>() {
            Ok(Value::Int(num))
        } else if let Some(value) = self.heap_table.get(s) {
            Ok(value.clone())
        } else {
            match s.to_lowercase().as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => Err(format!("Invalid value: {}", s)),
            }
        }
    }

    fn interpolate_string(&self, s: &str) -> Result<String, String> {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '$' && chars.peek() == Some(&'{') {
                chars.next(); // consume '{'
                let mut expr = String::with_capacity(32);
                let mut depth = 1;

                while let Some(c) = chars.next() {
                    if c == '{' {
                        depth += 1;
                    } else if c == '}' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    expr.push(c);
                }

                if !expr.is_empty() {
                    let value = self.eval(&self.parse(&expr)?)?;
                    result.push_str(&format!("{:?}", value));
                }
            } else {
                result.push(c);
            }
        }
        Ok(result)
    }

    fn parse_expr(&self, tokens: &[String]) -> Result<Exp, String> {
        let mut pos = 0;
        let capacity = tokens.len() / 2;
        let mut stack = Vec::with_capacity(capacity);
        let mut operators = Vec::with_capacity(capacity);

        while pos < tokens.len() {
            match &tokens[pos][..] {
                "(" => {
                    operators.push(tokens[pos].clone());
                }
                ")" => {
                    while let Some(op) = operators.last() {
                        if op == "(" {
                            operators.pop();
                            break;
                        }
                        Self::apply_operator(&mut stack, operators.pop().unwrap())?;
                    }
                }
                "+" | "-" | "*" | "/" | "^" | "%" | "and" | "or" | "xor" | "not" | "is" | "<="
                | ">=" | ">" | "<" | "in" => {
                    while let Some(op) = operators.last() {
                        if op == "(" || Self::precedence(op) < Self::precedence(&tokens[pos]) {
                            break;
                        }
                        Self::apply_operator(&mut stack, operators.pop().unwrap())?;
                    }
                    operators.push(tokens[pos].clone());
                }
                _ => {
                    stack.push(Exp::Val(tokens[pos].clone()));
                }
            }
            pos += 1;
        }

        while let Some(op) = operators.pop() {
            if op == "(" {
                return Err("Unclosed parenthesis".to_string());
            }
            Self::apply_operator(&mut stack, op)?;
        }

        if stack.len() == 1 {
            Ok(stack.pop().unwrap())
        } else {
            Err("Invalid expression".to_string())
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not() {
        let mut vm = VM::new();
        vm.set_value("a", Value::Bool(true));
        vm.set_value("b", Value::Bool(false));

        assert_eq!(vm.exec("a not b").unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_stmt() {
        let vm = VM::new();
        let stmt = vm
            .stmt("? and ?")
            .bind(Value::Bool(true))
            .bind(Value::Bool(false));
        assert_eq!(vm.exec_stmt(stmt).unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_stmt_string() {
        let vm = VM::new();
        let stmt = vm
            .stmt("? or ?")
            .bind(Value::String("hello".to_string()))
            .bind(Value::String("".to_string()));
        assert_eq!(vm.exec_stmt(stmt).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_stmt_number() {
        let vm = VM::new();
        let stmt = vm.stmt("? and ?").bind(Value::Int(1)).bind(Value::Int(0));
        assert_eq!(vm.exec_stmt(stmt).unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_basic_operations() {
        let mut vm = VM::new();
        vm.set_value("a", Value::Int(5));
        vm.set_value("b", Value::Int(10));

        assert_eq!(vm.exec("a <= b").unwrap(), Value::Bool(true));
        assert_eq!(vm.exec("b >= a").unwrap(), Value::Bool(true));
        assert_eq!(vm.exec("a < b").unwrap(), Value::Bool(true));
        assert_eq!(vm.exec("b > a").unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_value_parsing() {
        let vm = VM::new();
        assert_eq!(vm.parse_value("42").unwrap(), Value::Int(42));
        assert_eq!(
            vm.parse_value("\'hello\'").unwrap(),
            Value::String("hello".to_string())
        );
        assert_eq!(vm.parse_value("true").unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_string_interpolation() {
        let mut vm = VM::new();
        vm.set_value("x", Value::Int(42));
        assert_eq!(
            vm.interpolate_string("Value is ${x}").unwrap(),
            "Value is Int(42)"
        );
    }

    #[test]
    fn test_comparison_ops() {
        let mut vm = VM::new();
        vm.set_value("str1", Value::String("Hello World".to_string()));
        vm.set_value("str2", Value::String("World".to_string()));
        vm.set_value("num", Value::Int(5));

        assert_eq!("Hello World".contains("World"), true);
        assert_eq!(vm.exec("str2 in str1").unwrap(), Value::Bool(true));
        assert_eq!(vm.exec("num in str1").unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_basic_math() {
        let vm = VM::new();
        assert_eq!(vm.exec("2 + 3").unwrap(), Value::Int(5));
        assert_eq!(vm.exec("10 - 4").unwrap(), Value::Int(6));
        assert_eq!(vm.exec("3 * 4").unwrap(), Value::Int(12));
        assert_eq!(vm.exec("15 / 3").unwrap(), Value::Int(5));
        assert_eq!(vm.exec("7 % 3").unwrap(), Value::Int(1));
        assert_eq!(vm.exec("2 ^ 3").unwrap(), Value::Int(8));
    }

    #[test]
    fn test_order_of_operations() {
        let vm = VM::new();
        assert_eq!(vm.exec("2 + 3 * 4").unwrap(), Value::Int(14));
        assert_eq!(vm.exec("(2 + 3) * 4").unwrap(), Value::Int(20));
        assert_eq!(vm.exec("2 ^ 3 + 1").unwrap(), Value::Int(9));
        assert_eq!(vm.exec("10 - 2 * 3").unwrap(), Value::Int(4));
    }

    #[test]
    fn test_math_errors() {
        let vm = VM::new();
        assert!(vm.exec("5 / 0").is_err());
        assert!(vm.exec("10 % 0").is_err());
        assert!(vm.exec("2 ^ -1").is_err());
    }

    #[test]
    fn test_complex_math() {
        let mut vm = VM::new();
        vm.set_value("x", Value::Int(5));
        vm.set_value("y", Value::Int(3));

        assert_eq!(vm.exec("x * (y + 2)").unwrap(), Value::Int(25));
        assert_eq!(vm.exec("(x + y) * (x - y)").unwrap(), Value::Int(16));
        assert_eq!(vm.exec("x ^ 2 + y ^ 2").unwrap(), Value::Int(34));
    }
}
