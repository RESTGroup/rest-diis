# DIIS 外推 (基于 RSTSR 实现)

该程序库可通过 DIIS 算法，对计算化学或一些特定的迭代向量求解问题，减少迭代的步数。

## 使用方法

**需要注意！目前 API 不稳定，可能会作改动。**

```rust
// 简化的张量类型记号
type Tsr<T> = Tensor<T, DeviceOpenBLAS, IxD>;

// 默认的 DIIS 选项
let diis_flags = rstsr_diis::DIISIncoreFlagsBuilder::default().build().unwrap();
// 生成 DIIS 实例
let mut diis_driver = rstsr_diis::DIISIncore::<f64>::new(diis_flags, &device);

// 初猜向量 (譬如 SCF 下 Fock 矩阵 F_uv；CCSD 下的振幅 t1、t2)
let vec_init: Tsr<f64> = ...;
// 误差向量 (譬如某种定义下的 F_ai)
// 如果误差不好定义可以直接传 None (譬如 CCSD 的情形)
let err_init: Option<Tsr<f64>> = ...;
// 向 DIIS 实例插入初猜向量
// 参数表：
// - 初猜 (或待迭代) 向量
// - 初猜 (或待迭代) 误差 (有最好，没有可以传 None)
// - 迭代步数标记 (这里没有，建议传 None)
diis_driver.update(vec_init, err_init, None)

// 实际迭代
for niter in 0..max_iter {
    ...
    // 更新的待外推向量
    let vec_new: Tsr<f64> = ...;
    // 当前向量的误差
    let err_new: Option<Tsr<f64>> = ...;
    // 向 DIIS 实例插入待外推向量，并得到外推后的向量
    let vec_extrapolated = diis_driver.update(vec_new, err_new, None)
    ...
}
```
