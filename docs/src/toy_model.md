# 玩具模型上的示例

该文档是针对示例文件 [`toy_model.rs`](https://github.com/RESTGroup/rest-diis/blob/c19bfde9c8ab24636d5940119b85cb0d4f65126f/examples/toy_model.rs) 的补充说明。

## 1. 问题描述

该问题是一个简单的 4x4 矩阵求解问题：

$$
\mathbf{A} \bm{x} = \bm{b}
$$

一般来说，该问题是 $O(N^3)$ 复杂度的，因为该问题需要先对 $\mathbf{A}$ 分解为上三角、对角、下三角矩阵的乘积 (Cholesky 若对称半正定，SVD 或 QR 若非对称，等等)，将问题化为若干 $O(N^2)$ 三角矩阵求解问题；但矩阵的分解本身是 $O(N^3)$ 的。

现在，我们注意到 $\mathbf{A}$ 矩阵本身具有特征：其矩阵数值上，对角矩阵元相比于非对角元大许多。譬如示例中用到的矩阵
```rust
[
    [1.0, 0.1, 0.3, 0.2],
    [0.1, 1.5, 0.2, 0.1],
    [0.3, 0.2, 1.8, 0.2],
    [0.2, 0.1, 0.2, 1.3],
]
```
在这种情况下，我们记矩阵 $\mathbf{D} = \mathrm{diag} (\mathbf{A}) \simeq \mathbf{A}$ 是矩阵 $\mathbf{A}$ 的对角元构成的矩阵；其对角值向量是 $\bm{d}$。

这种具有对角值大特性的矩阵，是否有办法避免 $O(N^3)$ 复杂度的计算？DIIS 提供一种 (不保证一定求解成功，但在计算化学中屡试不爽) 的策略。

## 2. 简单迭代策略

在具体应用中，DIIS 迭代需要以简单迭代算法为基础。对于当前问题，简单迭代算法设计如下。

首先，初猜向量可以通过下述表达式给出：
$$
\bm{b} = \mathbf{A} \bm{x} \simeq \mathbf{D} \bm{x}
\; \Rightarrow \;
\bm{x} \simeq \mathbf{D}^{-1} \bm{b} = \bm{b} \div \bm{d}
$$
这里的除号是依元素除以的意思。因此，我们就以 $\bm{x}_0 = \mathbf{D}^{-1} \bm{b}$ 作为初猜。

其次，对于迭代过程中的向量 $\bm{x}_n$，为了得到下一次的向量 $\bm{x}_{n+1}$，我们注意并联立下述两式：
$$
\begin{align*}
    \bm{b}_n &= \mathbf{A} \bm{x}_n \\
    \bm{b} &= \mathbf{A} \bm{x}
\end{align*}
$$
得到
$$
\bm{x} = \mathbf{A}^{-1} (\bm{b} - \bm{b}_n) + \bm{x}_n \simeq \mathbf{D}^{-1} (\bm{b} - \bm{b}_n) + \bm{x}_n = (\bm{b} - \bm{b}_n) \div \bm{d} + \bm{x}_n
$$
我们就将该近似式定义为 $\bm{x}_{n + 1}$。这样就得到了简单迭代策略。

## 3. 简单迭代程序实现

程序实现如下。

{{#tabs }}
{{#tab name="Rust" }}
```rust
let f = |x: &Tsr<f64>| &a % x - &b;

// the initial guess
let mut x = &b / &d;
let mut x0;

// the number of iterations and the tolerance
let mut niter = 0;
let maxiter = 20;
let tol = 1e-7;

while niter < maxiter && f(&x).l2_norm_all() > tol {
    niter += 1;
    x0 = x;
    let b0 = &a % &x0;
    x = (&b - b0) / &d + &x0;
}
```
{{#endtab }}
{{#tab name="Python" }}
```python
def f(x):
    return a @ x - b

# the initial guess
x = b / d
x0 = np.zeros_like(x)

# the number of iterations and the tolerance
niter = 0
maxiter = 20
tol = 1e-7

while np.linalg.norm(f(x)) > tol and niter < maxiter:
    niter += 1
    x0, x = x, x0
    b0 = a @ x0
    x = (b - b0) / d + x0
```
{{#endtab }}
{{#endtabs }}


最终实际迭代次数是 19 次。

## 4. DIIS 实现

DIIS 相对于简单迭代情景，所需要增加的代码很少：
- 初始化 DIIS 实例需要 1--2 行；
- 将初猜提供给 DIIS 需要 1 行；
- 迭代过程中 DIIS 外推更新需要 1 行。

{{#tabs }}
{{#tab name="Rust" }}
```rust
let f = |x: &Tsr<f64>| &a % x - &b;

// the initial guess
let mut x = &b / &d;
let mut x0;

// DIIS incore driver
let diis_flags = DIISIncoreFlagsBuilder::default().build().unwrap(); // <==
let mut diis = DIISIncore::<f64>::new(diis_flags, &device);          // <==
// initial guess to DIIS
diis.update(x.to_owned(), None, None);                               // <==

// the number of iterations and the tolerance
let mut niter = 0;
let maxiter = 20;
let tol = 1e-7;

while niter < maxiter && f(&x).l2_norm_all() > tol {
    niter += 1;
    x0 = x;
    let b0 = &a % &x0;
    x = (&b - b0) / &d + &x0;
    x = diis.update(x, None, None);                                  // <==
}
```
{{#endtab }}
{{#tab name="Python" }}
```python
def f(x):
    return a @ x - b

# the initial guess
x = b / d
x0 = np.zeros_like(x)

# DIIS driver
diis = pyscf.lib.diis.DIIS()                                         ## <==
diis.update(x)                                                       ## <==

# the number of iterations and the tolerance
niter = 0
maxiter = 20
tol = 1e-7

while np.linalg.norm(f(x)) > tol and niter < maxiter:
    niter += 1
    x0, x = x, x0
    b0 = a @ x0
    x = (b - b0) / d + x0
    x = diis.update(x)                                               ## <==
```
{{#endtab }}
{{#endtabs }}

经过 DIIS 加速，该问题可以在 5 次迭代中完成。

## 5. 同时允许 semi-incore (disk I/O) 或 fully incore 实现的 DIIS

在实际的化学问题中，我们可能需要同时处理 semi-incore 与 fully incore 两种 DIIS：在大量计算的较小体系的情景下，我们希望用 fully incore DIIS 以增加效率；在计算较大体系时，则使用 semi-incore DIIS 以保证内存空间充足。

在这种情形下，`DIISAPI` 作为 dyn-compatible trait 提供了一种方式，以使得 semi-incore 与 fully incore 能同时处理：

```rust
// DIIS incore driver
let semi_incore = x.size() < 3; // assuming x > 3 requires semi-incore (disk I/O)
let mut diis: Box<dyn DIISAPI<Tsr<f64>>> = match semi_incore {
    false => Box::new({
        let diis_flags = DIISIncoreFlagsBuilder::default().build().unwrap();
        DIISIncore::<f64>::new(diis_flags, &device)
    }),
    true => Box::new({
        let diis_flags = DIISSemiIncoreFlagsBuilder::default().build().unwrap();
        DIISSemiIncore::<f64>::new(diis_flags, &device)
    }),
};
// initial guess to DIIS
diis.update(x.to_owned(), None, None);
```
