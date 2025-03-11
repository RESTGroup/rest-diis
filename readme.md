# DIIS 外推 (基于 RSTSR 实现)

该程序库可通过 DIIS 算法，对计算化学或一些特定的迭代向量求解问题，减少迭代的步数。

## 目前需要解决的问题

1. **硬盘存储**：这是主要问题。DIIS 需要存储 6-12 倍量的参数表；这对 SCF 的压力不大 (数倍于 $O(n_\mathrm{basis}^2)$ 的存储一般是可接受的)。但对于 CCSD 等问题，存储量会上升至数倍于 $O(n_\mathrm{occ}^2 n_\mathrm{vir}^2)$，最好能避免。
2. **浮点数**：这是次要问题。目前只支持 f64 类型的 DIIS。
3. **后端支持**：目前次要、但未来主要的问题。DIIS 的参数 (coefficients) 本身可以是任何后端的，毕竟其计算不耗时。但 DIIS 外推 (extrapolation) 则需要将向量输出到特定后端 (硬盘、CPU、GPU)。我们目前只支持 OpenBLAS 后端。
4. **任意张量维度类型**：这是次要问题。目前我们只接受动态维度 (`IxD`)，不接受固定维度输入。

## 使用方法

```rust
// 默认的 DIIS 选项
let diis_flags = rstsr_diis::DIISFlagsBuilder::default().build().unwrap();
// 生成 DIIS 实例
let mut diis_driver = rstsr_diis::DIIS::new(diis_flags, device);

// 初猜向量 (譬如 SCF 下 Fock 矩阵 F_uv；CCSD 下的振幅 t1、t2)
let vec_init: Tensor<f64, DeviceOpenBLAS, IxD> = ...;
// 误差向量 (譬如某种定义下的 F_ai)
// 如果误差不好定义可以直接传 None (譬如 CCSD 的情形)
let err_init: Option<Tensor<f64, DeviceOpenBLAS, IxD>> = ...;
// 向 DIIS 实例插入初猜向量
// 参数表：初猜向量，初猜误差 (这里没有)，迭代步数标记 (这里没有)
diis_driver.update(vec_init, err_init, None)

// 实际迭代
for niter in 0..max_iter {
    ...
    // 更新的待外推向量
    let vec_new: Tensor<f64, DeviceOpenBLAS, IxD> = ...;
    // 当前向量的误差
    let err_new: Option<Tensor<f64, DeviceOpenBLAS, IxD>> = ...;
    // 向 DIIS 实例插入待外推向量，并得到外推后的向量
    let vec_extrapolated = diis_driver.update(vec_new, err_new, None)
    ...
}
```
