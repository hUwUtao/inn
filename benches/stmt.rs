use criterion::{criterion_group, criterion_main, Criterion};
use inn::{Value, VM};
use std::hint::black_box;

fn run_expr(n: i32) -> Value {
    let mut vm = VM::new();
    vm.set_value("n", Value::Int(n));
    vm.set_value("str1", Value::String("Hello World".to_string()));
    vm.set_value("str2", Value::String("World".to_string()));
    vm.set_value("x", Value::Int(5));
    vm.set_value("y", Value::Int(3));
    let expr = vm
        .parse("(n <= 10 and str2 in str1) and (x * y <= 50)")
        .unwrap();
    vm.eval(&expr).unwrap()
}

fn run_stmt(n: i32) -> Value {
    let vm = VM::new();
    let stmt = vm
        .stmt("? <= ? and ? in ?")
        .bind(Value::Int(n))
        .bind(Value::Int(10))
        .bind(Value::String("World".to_string()))
        .bind(Value::String("Hello World".to_string()));
    vm.exec_stmt(stmt).unwrap()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("VM Operations");

    group.bench_function("complex expression", |b| b.iter(|| run_expr(black_box(15))));

    group.bench_function("prepared statement", |b| b.iter(|| run_stmt(black_box(15))));

    group.bench_function("sequence", |b| {
        b.iter(|| {
            let mut vm = VM::new();
            for i in 0..10 {
                vm.set_value(&format!("v{}", i), Value::Int(i));
                let expr = vm.parse(&format!("v{} <= 5", i)).unwrap();
                black_box(vm.eval(&expr).unwrap());
            }
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
