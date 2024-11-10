use inn::VM;

fn main() {
    let mut vm = VM::new();

    // Set sample values
    vm.set_value("first_name", inn::Value::String("John".to_string()));
    vm.set_value("last_name", inn::Value::String("Smith".to_string()));
    vm.set_value("age", inn::Value::Int(25));
    vm.set_value("city", inn::Value::String("New York".to_string()));
    vm.set_value("country", inn::Value::String("USA".to_string()));
    vm.set_value("is_active", inn::Value::Bool(true));
    vm.set_value("is_admin", inn::Value::Bool(false));
    vm.set_value("user_id", inn::Value::Int(1234));
    vm.set_value("email", inn::Value::String("john@example.com".to_string()));
    vm.set_value("phone", inn::Value::String("555-0123".to_string()));
    vm.set_value("title", inn::Value::String("Hello World".to_string()));
    vm.set_value("message", inn::Value::String("Hello".to_string()));
    vm.set_value("status", inn::Value::Bool(true));
    vm.set_value("description", inn::Value::String("test string".to_string()));
    vm.set_value("enabled", inn::Value::Bool(false));
    vm.set_value("valid", inn::Value::Bool(true));
    vm.set_value("username", inn::Value::String("johnsmith".to_string()));
    vm.set_value("price", inn::Value::Int(25));
    vm.set_value("quantity", inn::Value::Int(5));

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
            match vm.parse(expr) {
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
