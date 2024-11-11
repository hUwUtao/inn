use inn::{VMExec, Value, VM};
fn main() {
    let mut vm = VM::new();

    println!("Initializing virtual machine with test data...");

    // Create some test data
    vm.set_value("x", Value::Int(42));
    println!("Set variable 'x' = 42");

    vm.set_value("y", Value::Int(10));
    println!("Set variable 'y' = 10");

    vm.set_value("str2", Value::String("World".to_string()));
    println!("Set variable 'str2' = \"World\"");

    vm.set_value("flag", Value::Bool(true));
    println!("Set variable 'flag' = true");

    println!("\nParsing expression...");
    let expr = VM::parse(
        "(x * y + 5) ^ 2 and (str2 in 'Hello World {x + y * y} {y}') or (flag and y >= 10)",
    )
    .unwrap();
    println!("Successfully parsed expression into AST");

    println!("\nAST Structure:");
    println!("{:#?}", expr.clone());

    println!("\nCompiling to VM instructions...");
    let vmexec = VMExec::compile(expr.clone()).unwrap();
    println!("Successfully compiled to VM executable");

    println!("\nCompiled VM Instructions:");
    println!("{:#?}", vmexec);

    println!("\nExecuting compiled program...");
    let result = vm.exec_aot(&vmexec).unwrap();
    println!("Execution completed with result: {:?}", result);

    #[cfg(feature = "test-serde")]
    {
        println!("\nTesting serialization...");

        // Save as JSON
        let json = serde_json::to_string_pretty(&vmexec).unwrap();
        println!("Serialized to JSON format");
        std::fs::write("target/test.json", &json).unwrap();
        println!("Saved JSON to target/test.json");

        // Save as RON
        let ron = ron::ser::to_string_pretty(&vmexec, ron::ser::PrettyConfig::default()).unwrap();
        println!("Serialized to RON format");
        std::fs::write("target/test.ron", &ron).unwrap();
        println!("Saved RON to target/test.ron");

        // Deserialize and execute JSON
        let json_expr: inn::VMExec = serde_json::from_str(&json).unwrap();
        println!("Successfully deserialized from JSON");
        let json_result = vm.exec_aot(&json_expr).unwrap();
        println!("JSON execution result: {:?}", json_result);

        // Deserialize and execute RON
        let ron_expr: inn::VMExec = ron::de::from_str(&ron).unwrap();
        println!("Successfully deserialized from RON");
        let ron_result = vm.exec_aot(&ron_expr).unwrap();
        println!("RON execution result: {:?}", ron_result);
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile() {
        let mut vm = VM::new();
        vm.set_value("a", Value::Int(1));
        vm.set_value("b", Value::Int(2));

        let expr = VM::parse("a + b * 2").unwrap();
        let _vmexec = VMExec::compile(expr).unwrap();
    }

    #[cfg(feature = "test-serde")]
    #[test]
    fn test_serde() {
        let mut heap_table = std::collections::HashMap::new();
        VM::set_value(&mut heap_table, "x", Value::Int(42));

        let expr = VM::parse("x * 2").unwrap();
        let vmexec = VMExec::compile(expr).unwrap();

        let json = serde_json::to_string(&vmexec).unwrap();
        let deserialized: VMExec = serde_json::from_str(&json).unwrap();

        assert_eq!(vmexec, deserialized);

        let ron = ron::ser::to_string(&vmexec).unwrap();
        let ron_deserialized: VMExec = ron::de::from_str(&ron).unwrap();

        assert_eq!(vmexec, ron_deserialized);
    }
}
