use criterion::{criterion_group, criterion_main, Criterion};
use inn::{VMExec, Value, VM};
use std::hint::black_box;

fn run_expr(n: i32) -> Value {
    let mut vm = VM::new();
    vm.set_value("n", Value::Int(n));
    vm.set_value("str1", Value::String("Hello World".to_string()));
    vm.set_value("str2", Value::String("World".to_string()));
    vm.set_value("x", Value::Int(5));
    vm.set_value("y", Value::Int(3));

    let expr = VM::parse("(n <= 10 and str2 in str1) and (x * y <= 50)").unwrap();

    let exec = VMExec::compile(expr).unwrap();
    *vm.exec_aot(&exec).unwrap()
}

fn run_expr_jit(n: i32) -> Value {
    let mut vm = VM::new();
    vm.set_value("n", Value::Int(n));
    vm.set_value("str1", Value::String("Hello World".to_string()));
    vm.set_value("str2", Value::String("World".to_string()));
    vm.set_value("x", Value::Int(5));
    vm.set_value("y", Value::Int(3));

    let expr = VM::parse("(n <= 10 and str2 in str1) and (x * y <= 50)").unwrap();
    vm.eval(&expr).unwrap()
}

fn run_sequence_aot(vm: &VM, compiled_exprs: &[VMExec]) -> Vec<Value> {
    let mut results = Vec::with_capacity(10);
    for exec in compiled_exprs {
        results.push(*vm.exec_aot(exec).unwrap());
    }
    results
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("VM Operations");

    group.bench_function("complex expression (AOT)", |b| {
        b.iter(|| run_expr(black_box(15)))
    });

    group.bench_function("complex expression (JIT)", |b| {
        b.iter(|| run_expr_jit(black_box(15)))
    });

    group.bench_function("sequence", |b| {
        let mut vm = VM::new();
        let mut compiled_exprs = Vec::with_capacity(10);

        // Pre-compile expressions
        for i in 0..16 {
            vm.set_value(&format!("v{}", i), Value::Int(i));
            let expr = VM::parse(&format!("v{} <= 5", i)).unwrap();
            compiled_exprs.push(VMExec::compile(expr).unwrap());
        }

        b.iter(|| run_sequence_aot(&vm, &compiled_exprs))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
