# REST-DIIS 使用说明

## 1. 安装说明

该程序是库，没有主程序。若要在其他项目中使用该库，直接在其他库的 Cargo.toml 的 dependencies 中引入该库即可。

目前该程序没有发布到 crates.io 的计划。请将该库下载到本地。

该库不需要额外的编译；但在使用时，请作以下准备工作：
1. **准备 hdf5**。这是 cargo crate hdf5-metno 的要求；在 Ubuntu 系统中请执行
    ```bash
    sudo apt install libhdf5-dev
    ```
    对于不适合 apt install 的操作系统或集群环境，需要自行尝试解决。

2. **准备 OpenBLAS**。我们要求使用编译了 CBLAS、LAPACKE 的并行 OpenBLAS 后端 (pthread 或 OpenMP 均可)。并且您需要在项目中实现下述两者其一 (目的是将 libopenblas.so 和 OpenMP runtime 链入您的程序)：
    - `build.rs` 中引入如下语句：
      ```rust
      println!("cargo:rustc-link-lib=openblas");
      println!("cargo:rustc-link-lib=gomp"); // or other valid omp runtime library
      ```
    - `RUSTFLAGS = -l openblas -l gomp`

    <div class="warning">
    
    **这是 RSTSR 本身还有待发展导致的情况**

    必须要使用 OpenBLAS，是因为 RSTSR 目前还没有实现 Faer 后端的线性代数接口、或者 OpenBLAS 以外 BLAS 后端。
    
    DIIS 需要用到线性代数功能 (求本征值)，但计算性能需求并不大，所以理想的情况是不需要借助任何 FFI 后端实现 DIIS，即未来我们希望默认使用 Faer 后端。

    </div>

<div class="warning">

**Github Action 脚本不是理想的安装方式**

以 Github Action 为代表的持续集成/持续部署脚本，是用户说明文档之外的另一种重要资料，让用户学习如何安装和测试一个程序。

但对于 REST-DIIS 的 [Github Action](https://github.com/RESTGroup/rest-diis/blob/master/.github/workflows/test.yml) 而言，我们必须指出，这不是最好的安装方式。这是因为 Ubuntu 默认安装的 OpenBLAS (libopenblas-openmp-dev)，它并没有将 LAPACKE 同时编译；而 RSTSR 的线性代数功能需要使用到 LAPACKE。作为解决方案，链入 liblapacke-dev 不是好的选择。

我们不建议使用 RSTSR `DeviceOpenBLAS` 后端的同时，链接的 OpenBLAS 不具有 LAPACKE 的编译。

</div>

## 2. 使用说明

### 2.1 HDF5 临时文件及其路径

本项目的 `DIISSemiIncore` 将会在 DIIS 外推过程中使用 HDF5 硬盘存储空间。该空间是临时存储，一般来说用户是不需要获得的 (以后也许我们会实现 DIIS 文件存读，但目前这不是亟需实现的功能)。

DIIS 的临时文件路径由环境变量 `REST_TMPDIR` 或 `TMPDIR` 指定。`REST_TMPDIR` 优先级更高。

对于临时文件的声明周期问题：
- 如果程序正常退出，那么 DIIS 临时文件会自然地删除。
- 如果程序被用户终止或崩溃，DIIS 临时文件很有可能会存留在硬盘中。用户需要手动删除这些文件。

### 2.2 程序使用大致说明

下述不完整的代码是程序演示。该 DIIS 使用 `DIISIncore`，即所有存储均在内存进行。

若想查看一个可以工作的完整代码，我们也在一个 [简单的玩具问题示例](https://github.com/RESTGroup/rest-diis/blob/master/examples/toy_model.rs) 中展示了 DIIS 的应用 (参考 [后一节](./toy_model.md))。

```rust
// 简化的张量类型记号
type Tsr<T> = Tensor<T, DeviceOpenBLAS, IxD>;

// 默认的 DIIS 选项
let diis_flags = rstsr_diis::DIISIncoreFlagsBuilder::default().build().unwrap();
// 生成 DIIS 实例
let mut diis_driver = rstsr_diis::DIISIncore::<Tsr<f64>>::new(diis_flags, &device);

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

### 2.3 程序功能

- 同时支持全内存 (incore) 与半内存 (semi-incore) 功能。其中，**半内存**是指输入的、以及 DIIS 外推输出的向量，都是完全存于内存的；但 DIIS 内部的存储空间使用了硬盘作为媒介。
    - 全内存的 DIIS 类型是 `DIISIncore`，设置参数的构造程序类型是 `DIISIncoreFlagsBuilder`。
    - 半内存的 DIIS 类型是 `DIISSemiIncore`，设置参数的构造程序类型是 `DIISSemiIncoreFlagsBuilder`。
    - 目前我们还没有实现全磁盘 (outcore) 的功能。
    - 由于目前数学库 RSTSR 还不支持 GPU 等设备，因此目前还不具备异构功能。

- 重要的设置参数 (flags)：
    - 不同的 DIIS 示例具有不同的设置参数类型，这些参数类型目前还互不相通。
    - `space`：DIIS 空间大小。该设置一方面并非越大越好，另一方面该数值越大、DIIS 内部存储的向量也会越多，对内存或磁盘的需求会越紧张。特别是对 CCSD 等参数量较大的问题，不建议该值设置地太大。
    - `min_space`：DIIS 初始外推空间大小。该数值需要不小于 1。
    - `pop_strategy`：DIIS 弹出策略。DIIS 可以看作是一种队列 (queue)，其队列大小是 `space` 参数所决定的。一般来说 DIIS 是作为双向队列 (dqueue) 使用的，即先入先出 (`DIISPopStrategy::Iteration`)；但既然 DIIS 同时还需要计算输入向量的误差，那么我们也可以每次弹出误差最大的向量 (`DIISPopStrategy::ErrDiagonal`)。但由于如果最后一次输入的向量误差较大 (这种情况一般来说是收敛不太成功的情况)，弹出该向量容易导致外推出来的向量总是有较大误差；因而这种情形下，我们会退回到先入先出的策略上。
    - `chunk`：对于较长的向量，其计算是分批次进行的。该值对走硬盘的 DIIS 更为关键。需要注意，该值的设置并非越大越好，特别是硬盘占用空间小于磁盘的缓冲区域的情景。我们认为一个 chunk 的大小小于 1 MB (128 k FP64) 是比较合适的。
    - `scratch_path`：DIIS 缓存文件的文件夹路径。该选项仅对走硬盘的 DIIS 提供。

- Cargo feature
    - `sequential_io`：DIIS 计算误差重叠矩阵与外推时，不使用异步 I/O。我们在实现 DIIS 时，默认是使用异步 I/O 的，因此该选项一般不建议开启。

### 2.4 Undefined Behavior 注意事项

我们在使用 DIIS 时，一般仅使用其 `diis.update(vec, err, iter) -> extrapolated_vec` 函数。在使用该函数时，有一些情况我们不会仔细检查，有可能会导致 UB。这需要用户自己保证，包括：
1. 每次传入的 `vec`, `err` 的向量最好是一维 (动态维度 `IxD`) 向量，且必须保证每次传入的向量长度一致。
2. 如果你的计算中，有任何一次 `err` 无法给出 (即需要传 None)，那么请自始至终都在 `err` 参数中传入 None。`err` 传入有效值或 None 在整个程序中是两种处理逻辑。
3. 一般来说 `iter` 传入 None 即可，除非有特殊需求。DIIS 是双向队列时，具体要弹出哪个向量是通过迭代步数给出的；如果用户传入不合理的 `iter`，那么弹出的过程可能会出现问题。
