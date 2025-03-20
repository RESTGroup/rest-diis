# 所涉及到的编程实践


## 1. 异步 I/O

RSTSR-DIIS 在处理 semi-incore 情景时，使用标准库实现异步 I/O (以 [semi_incore_inner_dot](https://github.com/RESTGroup/rest-diis/blob/c19bfde9c8ab24636d5940119b85cb0d4f65126f/src/diis_semi_incore.rs#L405-L449) 函数为例)。

首先，由于我们用到的变量通常有生命周期，因此不能直接使用 `std::thread::spawn` 函数开辟新的线程。其解决方案是，需要使用 `std::thread::scope` 先圈住声明周期，随后进行双线程。

以下述伪代码为例。假设我们要执行 `func_a` 与 `func_b` (两者是 FnOnce，不返回任何结果)
```rust
for i in (0..niter) {
    func_a();
    func_b();
}
```
实际情景中，`func_b` 会依赖于 `func_a`，即不能乱序执行；但 `func_b` 不依赖于 `func_a`。若现在要进行异步问题，我们可以这么操作：
```rust
std::thread::scope(|scope| {  // 圈住生命周期
    let mut task = scope.spawn(|| {});
    for i in (0..niter) {
        func_a();             // 此时 i 步的 func_a 与 i-1 步的 func_b 同时进行
        task.join().unwrap(); // 相当于 barrier，保证现在执行 func_b 时已经完成 func_a
        task = scope.spawn(move || {
            func_b();         // func_b 并不在 master thread 上生成，而是平行进行的
        })
    }
})
```

很可能我们在 `func_b` 时需要用到一些可变引用；碰到这种情况会比较麻烦，需要在 spawn 前传入不可变引用，然后在 spawn 内部用一些 unsafe 技巧弄成可变的。这其实有点像 OpenMP 并行时，一些变量是 shared 即所有线程可见的；这在 Rust 中其实是 unsafe 的，但这样的程序非常好写，而且大多数时候不会真的造成 race，那我的看法是不如就大方地用 unsafe (= trust me)。

同时，我们指出上面的做法是 post-process 的异步。另一种异步策略是 prefetch：
```python
# 伪代码
func_a(0)
for i in range(1, niter)
    barrier()
    if i < niter - 1:
        task = spawn(func_a(i + 1))  # prefetch
    func_b(i)
```
但 prefetch 的策略我感觉不是非常好写，因为会对第一次循环读取需要在循环外进行、会增加一些判断语句去确定是否有下一轮循环。

被异步到分支线程的任务不一定必须是 I/O，也可以是计算过程本身。
