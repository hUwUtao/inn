use inn::Value::{Bool, Int, String};
use inn::VM;

fn main() {
    let mut vm = VM::new();

    // Set sample values
    vm.set_value("first_name", String("John".to_string()));
    vm.set_value("last_name", String("Smith".to_string()));
    vm.set_value("age", Int(25));
    vm.set_value("city", String("New York".to_string()));
    vm.set_value("country", String("USA".to_string()));
    vm.set_value("is_active", Bool(true));
    vm.set_value("is_admin", Bool(false));
    vm.set_value("user_id", Int(1234));
    vm.set_value("email", String("john@example.com".to_string()));
    vm.set_value("phone", String("555-0123".to_string()));
    vm.set_value("title", String("Hello World".to_string()));
    vm.set_value("message", String("Hello".to_string()));
    vm.set_value("status", Bool(true));
    vm.set_value("description", String("test string".to_string()));
    vm.set_value("enabled", Bool(false));
    vm.set_value("valid", Bool(true));
    vm.set_value("username", String("johnsmith".to_string()));
    vm.set_value("price", Int(25));
    vm.set_value("quantity", Int(5));

    // Complex expressions with various operators and parentheses
    let expressions = vec![
        "(is_active and is_admin) or (status and enabled)",
        "((user_id > 5) and (price <= 10)) or (quantity is true)",
        "is_active in 'Hello World' and (price >= 42)",
        "(title in message) xor (first_name not last_name)",
        "(is_active or is_admin) and (status xor enabled) or (valid in description)",
        "((user_id <= price) or (price >= quantity)) and (is_active in is_admin)",
        "(true and false) xor (42 > 10)",
        "(username in 'John Doe') and (age >= 18)",
        "((is_active and is_admin) or (status and enabled)) xor ((valid or description) and (enabled or valid))",
        "(user_id not price) and (title in 'test') or (price <= 100)",
        "(quantity <= 10 and message in title) and (user_id * price <= 50)",
    ];

    for expr in expressions {
        #[cfg(debug_assertions)]
        {
            println!("\nExpression: {}", expr);
            match VM::parse(expr) {
                Ok(ast) => {
                    println!("AST Tree:");
                    println!("{:#?}", ast);
                    match vm.eval(&ast) {
                        Ok(result) => println!("Evaluation Result: {:?}", result),
                        Err(e) => println!("Evaluation Error: {}", e),
                    }
                }
                Err(e) => println!("Parse Error: {}", e),
            }
            println!("----------------------------------------");
        }
        #[cfg(not(debug_assertions))]
        {
            // match vm.exec(expr) {
            // Ok(result) => println!("Result: {:?}", result),
            // Err(e) => println!("Error: {}", e),
            // }
            let _ = vm.exec(expr);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not() {
        let mut vm = VM::new();
        vm.set_value("a", Bool(true));
        vm.set_value("b", Bool(false));

        assert_eq!(vm.exec("a not b").unwrap(), Bool(true));
    }

    // #[test]
    // fn test_stmt() {
    //     let vm = VM::new();
    //     let stmt = vm.stmt("? and ?").bind(Bool(true)).bind(Bool(false));
    //     assert_eq!(vm.exec_stmt(stmt).unwrap(), Bool(false));
    // }

    // #[test]
    // fn test_stmt_string() {
    //     let vm = VM::new();
    //     let stmt = vm
    //         .stmt("? or ?")
    //         .bind(String("hello".to_string()))
    //         .bind(String("".to_string()));
    //     assert_eq!(vm.exec_stmt(stmt).unwrap(), Bool(true));
    // }

    // #[test]
    // fn test_stmt_number() {
    //     let vm = VM::new();
    //     let stmt = vm.stmt("? and ?").bind(Int(1)).bind(Int(0));
    //     assert_eq!(vm.exec_stmt(stmt).unwrap(), Bool(false));
    // }

    #[test]
    fn test_basic_operations() {
        let mut vm = VM::new();
        vm.set_value("a", Int(5));
        vm.set_value("b", Int(10));

        assert_eq!(vm.exec("a <= b").unwrap(), Bool(true));
        assert_eq!(vm.exec("b >= a").unwrap(), Bool(true));
        assert_eq!(vm.exec("a < b").unwrap(), Bool(true));
        assert_eq!(vm.exec("b > a").unwrap(), Bool(true));
    }

    #[test]
    fn test_string_interpolation() {
        let mut vm = VM::new();
        vm.set_value("name", String("World".to_string()));
        vm.set_value("num", Int(42));

        assert_eq!(
            vm.exec("'Hello {name}!'").unwrap(),
            String("Hello World!".to_string())
        );

        assert_eq!(
            vm.exec("'The answer is {num}'").unwrap(),
            String("The answer is 42".to_string())
        );
    }

    #[test]
    fn test_comparison_ops() {
        let mut vm = VM::new();
        vm.set_value("str1", String("Hello World".to_string()));
        vm.set_value("str2", String("World".to_string()));
        vm.set_value("num", Int(5));

        assert_eq!("Hello World".contains("World"), true);
        assert_eq!(vm.exec("str2 in str1").unwrap(), Bool(true));
        assert_eq!(vm.exec("num in str1").unwrap(), Bool(false));
    }

    #[test]
    fn test_basic_math() {
        let vm = VM::new();
        assert_eq!(vm.exec("2 + 3").unwrap(), Int(5));
        assert_eq!(vm.exec("10 - 4").unwrap(), Int(6));
        assert_eq!(vm.exec("3 * 4").unwrap(), Int(12));
        assert_eq!(vm.exec("15 / 3").unwrap(), Int(5));
        assert_eq!(vm.exec("7 % 3").unwrap(), Int(1));
        assert_eq!(vm.exec("2 ^ 3").unwrap(), Int(8));
    }

    #[test]
    fn test_order_of_operations() {
        let vm = VM::new();
        assert_eq!(vm.exec("2 + 3 * 4").unwrap(), Int(14));
        assert_eq!(vm.exec("(2 + 3) * 4").unwrap(), Int(20));
        assert_eq!(vm.exec("2 ^ 3 + 1").unwrap(), Int(9));
        assert_eq!(vm.exec("10 - 2 * 3").unwrap(), Int(4));
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
        vm.set_value("x", Int(5));
        vm.set_value("y", Int(3));

        assert_eq!(vm.exec("x * (y + 2)").unwrap(), Int(25));
        assert_eq!(vm.exec("(x + y) * (x - y)").unwrap(), Int(16));
        assert_eq!(vm.exec("x ^ 2 + y ^ 2").unwrap(), Int(34));
    }
}
