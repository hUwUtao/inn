use inn::{VMExec, Value, VM};
fn main() {
    let mut vm = VM::new();

    // Create some test data
    vm.set_value("x", Value::Int(42));
    vm.set_value("y", Value::Int(10));
    vm.set_value("str2", Value::String("World".to_string()));
    vm.set_value("flag", Value::Bool(true));

    let expr = vm
        .parse("(x * y + 5) ^ 2 and (str2 in 'Hello World') or (flag and y >= 10)")
        .unwrap();

    let vmexec = VMExec::compile(expr.clone()).unwrap();

    println!("{:#?}\n{:#?}", expr.clone(), vmexec);

    #[cfg(feature = "test-serde")]
    {
        // Save as JSON
        let json = serde_json::to_string_pretty(&vmexec).unwrap();
        std::fs::write("target/test.json", &json).unwrap();

        // Save as RON
        let ron = ron::ser::to_string_pretty(&vmexec, ron::ser::PrettyConfig::default()).unwrap();
        std::fs::write("target/test.ron", &ron).unwrap();

        // Deserialize and execute JSON
        let _json_expr: inn::VMExec = serde_json::from_str(&json).unwrap();
        // println!("{:#?}", json_expr);

        // Deserialize and execute RON
        let _ron_expr: inn::VMExec = ron::de::from_str(&ron).unwrap();
        // println!("{:#?}", ron_expr);
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

        let expr = vm.parse("a + b * 2").unwrap();
        let _vmexec = VMExec::compile(expr).unwrap();
    }

    #[cfg(feature = "test-serde")]
    #[test]
    fn test_serde() {
        let mut vm = VM::new();
        vm.set_value("x", Value::Int(42));

        let expr = vm.parse("x * 2").unwrap();
        let vmexec = VMExec::compile(expr).unwrap();

        let json = serde_json::to_string(&vmexec).unwrap();
        let deserialized: VMExec = serde_json::from_str(&json).unwrap();

        assert_eq!(vmexec, deserialized);

        let ron = ron::ser::to_string(&vmexec).unwrap();
        let ron_deserialized: VMExec = ron::de::from_str(&ron).unwrap();

        assert_eq!(vmexec, ron_deserialized);
    }
}
