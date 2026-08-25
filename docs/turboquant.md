# TurboQuant 算法原理、Codebook、Scale、QJL 与 vLLM 实现

本文从算法原理出发，回答下面几个最容易混淆的问题：

1. 随机正交旋转后，到底什么东西是“均匀”的？
2. 单个 coordinate 为什么是 Beta 型分布，高维下又为什么近似 Gaussian？
3. 不同 coordinate 是严格独立，还是只在高维下近似独立？
4. Lloyd-Max codebook 是从真实 KV 数据训练出来的吗？
5. centroid、midpoint、K norm、V scale、V zero 分别是什么？
6. TurboQuant-MSE 与 TurboQuant-Prod 有什么区别？
7. QJL 是不是 scale？vLLM 为什么没有使用它？
8. 本项目的 `turboquant_4bit_nc` 到底在每个 cache slot 中保存什么？

本文区分三个层次，避免把不同实现混在一起：

- **TurboQuant 论文算法**：随机正交旋转、MSE quantizer、QJL residual；
- **当前 vLLM TurboQuant backend**：Hadamard rotation、Lloyd-Max MSE K、
  uniform V、norm correction，明确省略 QJL；
- **本项目**：固定 Qwen3-4B shape 的 `turboquant_4bit_nc` CUDA Decode
  Stage1/Stage2 优化，读取与 vLLM SoA Store 相同的压缩布局。

---

## 1. 先给出最直接的答案

### 1.1 旋转后得到的是均匀分布吗？

要分清“整个向量”和“单个坐标”。

给定一个单位向量 $x\in\mathbb{R}^d$，如果 $R$ 是 Haar 随机正交矩阵：

$$
y=Rx,
$$

那么 **整个向量 $y$ 在单位球面 $S^{d-1}$ 上均匀分布**。但这不代表
每个 $y_i$ 在某个区间上服从 uniform distribution。

单个 coordinate $y_i$ 是一个集中在 0 附近的对称 Beta 型分布。维度越高，
它越集中；经过适当缩放后才近似 Gaussian：

$$
\sqrt dy_i \xrightarrow[d\to\infty]{\text{distribution}} \mathcal N(0,1),
$$

也就是：

$$
y_i \approx \mathcal N(0,1/d).
$$

所以正确表述是：

> 随机旋转让整个单位向量在球面上均匀；单坐标服从对称 Beta 型边缘分布，
> 高维时近似 $\mathcal N(0,1/d)$，而不是单坐标变成 uniform distribution。

### 1.2 Codebook 怎么构建？需要采样真实 KV 吗？

本项目摘取的 vLLM 实现 **不采样真实 KV，也不运行 K-means**。它直接采用：

$$
X\sim\mathcal N(0,1/d)
$$

作为旋转后 coordinate 的工程近似分布，然后对这个已知 Gaussian PDF 做数值
积分，迭代求解 Lloyd-Max 条件，得到 $2^b$ 个 centroid。

对于 `head_dim=128, bits=4`：

```text
variance   = 1 / 128
centroids  = 16 个 FP32 数
midpoints  = 15 个 FP32 决策边界
```

Codebook 只依赖 $d$ 和 bit width，不依赖模型层、token 或真实 KV 数据。

### 1.3 Scale 是怎么来的？

本实现中没有一个统一的“TurboQuant scale”。至少要区分四个量：

| 名称 | 用途 | 如何得到 | 存储 |
| --- | --- | --- | --- |
| K norm | 恢复 K 向量幅度 | $\lVert K\rVert_2$，NC 时再除以 centroid-vector norm | 每 token/KV head 一个 FP16 |
| V scale | V 的 affine quantization 步长 | $(v_{max}-v_{min})/15$ | 每 token/KV head 一个 FP16 |
| V zero | V affine quantization 的起点 | $v_{min}$ | 每 token/KV head 一个 FP16 |
| Attention scale | scaled dot-product attention | $1/\sqrt d$ | Kernel 常量/参数，不属于 KV 量化 metadata |

K 的标量更准确地叫 **norm**，不是普通 min-max quantization scale。

### 1.4 QJL 是 scale 吗？

不是。QJL 是 **1-bit Quantized Johnson-Lindenstrauss residual encoding**。
TurboQuant-Prod 先用较低 bit 的 MSE quantizer 得到近似向量，再对残差做随机
投影并保存 sign bit，用来构造无偏 inner-product estimator。

QJL 与 scale 的区别：

```text
scale / norm：恢复数值幅度
QJL：编码 MSE quantization residual 的方向信息，修正内积偏差
```

当前 vLLM TurboQuant backend 明确省略 QJL；本项目也没有 QJL 数据、QJL
projection 或 QJL correction。

### 1.5 本项目是不是“4 bit 加一个 float scale”？

不完全正确。对 `D=128` 的每个 token/KV head：

```text
K: 128 * 4 bit = 64 B indices + 1 个 FP16 corrected norm
V: 128 * 4 bit = 64 B indices + 1 个 FP16 scale + 1 个 FP16 zero
```

因此逻辑 slot 是：

```text
64 B + 64 B + 2 B + 2 B + 2 B = 134 B
```

另外还有一份由所有 token 共享的 16-entry FP32 centroid table。不是“每个向量
4 bit 加一个 FP32 scale”，也不是每个 coordinate 各有一个 scale。

---

## 2. 为什么高维旋转能把 Vector Quantization 化成 Scalar Quantization

### 2.1 从归一化开始

先取一个非零向量 $k\in\mathbb R^d$：

$$
\gamma=\lVert k\rVert_2,\qquad x=\frac{k}{\gamma}.
$$

此时 $x$ 位于单位球面：

$$
\lVert x\rVert_2=1.
$$

归一化的目的，是把“方向”和“长度”拆开：

- 方向交给 rotation + scalar codebook；
- 长度 $\gamma$ 作为一个标量单独保存。

如果不先归一化，不同 token 的 K 幅度差异会让同一份固定 codebook 很难同时
覆盖所有范围。

### 2.2 随机正交旋转保持哪些东西？

正交矩阵满足：

$$
R^TR=I.
$$

所以它保持范数：

$$
\lVert Rx\rVert_2^2=x^TR^TRx=\lVert x\rVert_2^2.
$$

也保持两个向量的内积：

$$
(Rx)^T(Rz)=x^Tz.
$$

旋转本身不丢信息，也不会改变精确 Attention score。误差来自后面的低比特
quantization，而不是正交变换。

### 2.3 为什么说旋转后在球面上均匀？

如果 $R$ 从所有正交矩阵上的 Haar distribution 随机抽取，那么对任意固定
单位向量 $x$，$Rx$ 的方向没有偏好。对任何另一个正交矩阵 $U$：

$$
URx
$$

与 $Rx$ 同分布。这种 rotation invariance 唯一对应单位球面上的均匀分布。

这里“均匀”描述的是整个 $d$ 维方向，而不是每一维在 $[-1,1]$ 上均匀。

### 2.4 单 coordinate 的精确分布是什么？

设 $Y=(Y_1,\ldots,Y_d)$ 均匀分布在 $S^{d-1}$。单个 $Y_i$ 的 density 为：

$$
f_d(t)=
\frac{\Gamma(d/2)}{\sqrt\pi\Gamma((d-1)/2)}
(1-t^2)^{(d-3)/2},\qquad -1\le t\le1.
$$

它是以 0 为中心的对称分布。常见的两个等价 Beta 表述是：

$$
Y_i^2\sim\mathrm{Beta}\left(\frac12,\frac{d-1}{2}\right),
$$

以及：

$$
\frac{Y_i+1}{2}
\sim\mathrm{Beta}\left(\frac{d-1}{2},\frac{d-1}{2}\right).
$$

因此“coordinate 服从 Beta distribution”是简写。严格说 coordinate 在
$[-1,1]$ 上服从 **对称 Beta 型分布**；它的平方才直接服从上面的 Beta。

### 2.5 为什么高维下近似 Gaussian？

可以用一个标准构造理解均匀球面向量。令：

$$
G_1,\ldots,G_d\overset{iid}{\sim}\mathcal N(0,1),
$$

则：

$$
Y=\frac{G}{\lVert G\rVert_2}
$$

在单位球面上均匀。高维下根据大数定律：

$$
\frac{\lVert G\rVert_2}{\sqrt d}\to1.
$$

所以：

$$
Y_i=\frac{G_i}{\lVert G\rVert_2}
\approx\frac{G_i}{\sqrt d}
\sim\mathcal N(0,1/d).
$$

这就是 vLLM 在 $d\ge64$ 时采用 Gaussian approximation 的直观来源。

### 2.6 Coordinate 之间真的独立吗？

不是严格独立，因为它们必须满足：

$$
\sum_{i=1}^dY_i^2=1.
$$

知道很多坐标的平方后，剩余坐标能量必然受约束。它们是 uncorrelated/symmetric，
但不能由此推出严格 independence。

高维下，对于固定数量的坐标：

$$
(\sqrt dY_1,\ldots,\sqrt dY_m)
$$

会联合趋近 $m$ 个独立的标准 Gaussian。这种 **有限坐标的渐近近独立性**
使逐 coordinate scalar quantization 成为有效近似。

### 2.7 这为什么能简化 Vector Quantization？

通用 $d$ 维 vector quantizer 若每维使用 $b$ bit，可能需要规模接近
$2^{bd}$ 的 codebook，无法在线查询。旋转后每个 coordinate 具有近似相同
的已知边缘分布，并在高维下近似独立，于是可以复用同一个一维 quantizer：

```text
一个 d 维巨大 codebook
        ↓
d 次同一个 2^b-entry scalar codebook lookup
```

它不是说 scalar quantization 与最优 vector quantization 完全等价，而是利用
随机旋转后的统计结构，获得 data-oblivious、在线可执行且失真接近最优的方案。

---

## 3. 论文随机旋转与 vLLM Hadamard Rotation 的区别

### 3.1 论文层面的随机正交矩阵

TurboQuant 理论分析使用随机正交旋转。随机性让任意固定输入方向转成球面均匀
方向，从而可以严格推导 Beta 边缘分布和近独立性质。

生成完整 Haar orthogonal matrix 并做 dense $d\times d$ matmul 成本不低，
工程实现通常使用 structured orthogonal transform，例如 randomized Hadamard。

### 3.2 当前 vLLM 实际使用什么？

当前摘取的 vLLM backend 使用归一化 Sylvester Hadamard matrix $H$：

$$
H^TH=I,\qquad H=H^T,\qquad H^{-1}=H.
$$

它在初始化时构造 $D\times D$ 的 FP32 matrix，Store 中通过：

```python
x_hat = k_flat / (norms + 1e-8)
y = x_hat @ PiT
```

完成 K rotation。Decode 对 Q 使用匹配的 rotation。

需要注意：当前代码注释明确说明它使用 **pure Hadamard**，没有额外 random sign。
原因是 codebook 关于 0 对称，单纯 sign flip 会映射到镜像 centroid，工程上没有
观察到 quantization-quality 收益。

### 3.3 Pure Hadamard 后能否严格声称 coordinate 是 Beta/Gaussian？

不能无条件严格声称。Pure deterministic Hadamard 对任意固定输入并不产生 Haar
球面均匀分布；例如某些特殊输入可能与 Hadamard row/column 高度对齐。

因此应区分：

- **论文理论**：Haar/random orthogonal rotation 导出 Beta 与 Gaussian 近似；
- **vLLM 工程**：用 Hadamard 做便宜、正交、强混合的 transform，并继续复用
  对 $\mathcal N(0,1/d)$ 优化的 codebook；
- **实际效果**：依赖模型 K 分布、Hadamard mixing、norm correction 和实测质量。

面试时不要把理论随机旋转的精确分布结论原封不动地说成 deterministic Hadamard
对所有输入都严格成立。

### 3.4 为什么 Q 也必须旋转？

假设 Store 保存的是：

$$
k_r=kH.
$$

Decode 时令：

$$
q_r=qH.
$$

因为 $HH^T=I$：

$$
q_rk_r^T=qHH^Tk^T=qk^T.
$$

本项目直接在 rotated space 计算 QK，不需要先把 K inverse-rotate 回原空间。
Q rotation 可以在 launcher 或 attention prologue 完成，但必须计清性能边界。

---

## 4. Lloyd-Max Codebook 到底怎样构造

### 4.1 优化目标

给定 scalar random variable $X\sim f(x)$，希望用 $L=2^b$ 个重建值
$c_0,\ldots,c_{L-1}$ 使 MSE 最小：

$$
D=\mathbb E[(X-Q(X))^2].
$$

Quantizer 将实数轴划分成 $L$ 个区间 $I_i=[a_i,a_{i+1})$，落入区间
$I_i$ 的值重建为 $c_i$。

### 4.2 Lloyd-Max 的两个必要条件

对于 squared error，固定 centroids 时，最近邻决策边界是相邻 centroid 中点：

$$
a_i=\frac{c_{i-1}+c_i}{2}.
$$

固定区间时，使区间内 MSE 最小的 centroid 是条件均值：

$$
c_i=\mathbb E[X\mid X\in I_i]
=\frac{\int_{a_i}^{a_{i+1}}x f(x)dx}
{\int_{a_i}^{a_{i+1}}f(x)dx}.
$$

Lloyd-Max iteration 就是在这两个条件之间交替更新，直到 centroid 变化足够小。

### 4.3 vLLM 使用的目标 PDF

本地 `centroids.py` 直接设置：

$$
\sigma^2=1/d,
$$

$$
f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{x^2}{2\sigma^2}\right).
$$

也就是说，它没有：

- 从真实 K 中收集 calibration dataset；
- 从 Beta/Gaussian 中 Monte Carlo sampling；
- 对样本运行 K-means。

它直接对 analytic Gaussian PDF 做数值积分。

### 4.4 本地实现的具体迭代过程

对于 $d=128,b=4$：

1. 令 $L=16,\sigma=1/\sqrt{128}$；
2. 在 $[-3.5\sigma,3.5\sigma]$ 内均匀初始化 16 个 centroid；
3. 计算相邻 centroid midpoint；
4. 对每个区间用 trapezoidal integration 计算概率质量和一阶矩；
5. 用条件均值更新 centroid；
6. 最多迭代 200 次，最大变化小于 $10^{-10}$ 时停止；
7. 返回 16 个 FP32 centroid 和 15 个 midpoint；
8. `get_centroids` 按 `(d,bits)` 缓存结果。

伪代码为：

```python
centroids = initialize_evenly()

for _ in range(max_iter):
    boundaries = midpoint(centroids[:-1], centroids[1:])
    for interval in intervals(boundaries):
        probability = integrate(pdf, interval)
        first_moment = integrate(x * pdf(x), interval)
        new_centroid = first_moment / probability
    if max_abs(new_centroids - centroids) < tolerance:
        break
    centroids = new_centroids
```

### 4.5 Centroid 和 midpoint 分别有什么用？

- **Midpoint**：Store 量化时做 bucketize，决定一个 coordinate 属于哪个 index；
- **Centroid**：Decode 时根据 index 查表，得到该 coordinate 的重建值。

对于 4 bit：

```text
15 midpoints -> 将实数轴分成 16 个区间
4-bit index  -> 表示区间编号 0..15
16 centroids -> 每个区间的 reconstruction value
```

### 4.6 查“最近 centroid”如何实现？

centroid 已排序，squared-error 最近邻边界就是 midpoint。Store 不必逐个计算
16 个距离，只需在 15 个 midpoint 中二分搜索。

本项目摘取的 Store 对 4-bit 情况展开 4 轮 binary-search update，得到 index，
再将相邻两个 index 打包进一个 byte：

```text
low nibble  = dimension 2j
high nibble = dimension 2j+1
```

Decode 再执行反向 nibble unpack。

### 4.7 Codebook 是“训练参数”吗？

不是模型训练参数，也不需要 fine-tuning。它是由 `(head_dim, bits, assumed PDF)`
确定的 quantizer 常量。vLLM 在 backend 初始化时生成并缓存，再上传 GPU；所有
layer/token 可以共享同一组 centroid 数值。

如果目标 coordinate 分布明显偏离假设，固定 codebook 的实际 MSE 可能不再
最优，但它换来了 data-oblivious、无需 calibration 和低初始化成本。

---

## 5. K 的完整 TurboQuant-MSE Store 流程

### 5.1 Step 1：保存向量幅度

对每个 token、每个 KV head：

$$
\gamma=\lVert k\rVert_2.
$$

先归一化：

$$
x=\frac{k}{\gamma+\epsilon}.
$$

### 5.2 Step 2：Hadamard rotation

$$
y=xH.
$$

此时仍有：

$$
\lVert y\rVert_2=1.
$$

### 5.3 Step 3：逐 coordinate bucketize

对每个 $y_j$，用 midpoint 找 index：

$$
i_j=Q_{index}(y_j),\qquad i_j\in\{0,\ldots,15\}.
$$

对应重建 coordinate：

$$
c_j=C[i_j].
$$

整个 centroid-vector 是：

$$
c=(C[i_1],\ldots,C[i_d]).
$$

### 5.4 Step 4：Norm correction

理想的 $y$ 是单位向量，但量化后的 $c$ 一般不满足 $\lVert c\rVert_2=1$。
如果直接重建：

$$
\hat y=\gamma c,
$$

则：

$$
\lVert\hat y\rVert_2=\gamma\lVert c\rVert_2,
$$

会引入额外 norm distortion。

`turboquant_4bit_nc` 开启 norm correction，Store 实际保存：

$$
\gamma_{stored}=\frac{\gamma}{\lVert c\rVert_2}.
$$

Decode 重建：

$$
\hat y=\gamma_{stored}c,
$$

于是：

$$
\lVert\hat y\rVert_2=\gamma.
$$

这里恢复的是原 K 的 norm，但方向仍有 scalar quantization error。

### 5.5 为什么把 correction 放到 Store？

如果 Decode 每次读取这个历史 K 都重新计算 $\lVert c\rVert_2$，同一 token
会被反复付费。Store 只计算一次并把 correction 折叠进 FP16 norm，Decode
热路径只做：

```text
index -> centroid -> multiply stored_norm
```

这也是 `nc` 后缀代表的核心内容：norm correction，而不是额外 QJL。

### 5.6 Step 5：Bit packing 与 metadata store

对 `D=128,b=4`：

```text
128 indices * 4 bit = 512 bit = 64 B
corrected norm       = FP16    = 2 B
```

SoA layout 将 64 B index 放在 data region，把 FP16 norm 放到当前 block 的
K-norm metadata array。

---

## 6. K 的 Decode 与 Attention Score

### 6.1 Decode 如何重建 K tile？

对 packed byte：

```text
idx_lo = byte & 0xF
idx_hi = byte >> 4
```

查表并乘 norm：

```text
k[2j]   = centroid[idx_lo] * stored_norm
k[2j+1] = centroid[idx_hi] * stored_norm
```

本项目 V6-V9 每次使用一个对齐 `uint32` 读取四个 packed byte，即八个 K
coordinate，再通过 `half2` 写入 tile Shared Memory。

### 6.2 为什么不 inverse-rotate K？

因为 Q 已使用相同正交矩阵旋转。可直接计算：

$$
q_r\hat k_r^T.
$$

这与先将 K inverse-rotate 再和原始 Q 点积在数学上等价，但避免为每个历史 K
执行 inverse transform。

### 6.3 Attention scale 与 K norm 不是同一个量

Score 为：

$$
s=\frac{q_r\hat k_r^T}{\sqrt d}.
$$

其中：

- `stored_norm` 恢复每个 K token/head 的幅度；
- $1/\sqrt d$ 是所有 score 的 Attention scaling。

本项目 `D=128`，所以 Attention scale 是：

$$
1/\sqrt{128}\approx0.08838835.
$$

不要把它称为 TurboQuant quantization scale。

---

## 7. V 的 4-bit Uniform Quantization 与 Scale

### 7.1 为什么 K 和 V 使用不同 quantizer？

K 直接参与 QK inner product，方向和内积结构非常关键，因此使用 rotation +
Lloyd-Max。V 在 softmax 权重确定后参与 weighted sum，当前 vLLM 工程路径使用
更简单的 per-token/per-head min-max uniform quantization。

这是一项实现选择，不代表 V 在理论上永远不适合 TurboQuant-MSE 或其他
quantizer。

### 7.2 V scale 如何计算？

对一个 token/head 的 $D$ 个 V coordinate：

$$
v_{min}=\min_jv_j,\qquad v_{max}=\max_jv_j.
$$

4-bit 有 16 个 level，index 范围是 0 到 15：

$$
s_v=\max\left(\frac{v_{max}-v_{min}}{15},10^{-8}\right).
$$

量化：

$$
q_j=\mathrm{clip}\left(
\mathrm{round}\left(\frac{v_j-v_{min}}{s_v}\right),0,15
\right).
$$

Store 中的 `+0.5` 再转 integer 实现对非负归一化值的 round-to-nearest。

### 7.3 V zero 是什么？

本实现保存：

$$
z_v=v_{min}.
$$

重建：

$$
\hat v_j=q_js_v+z_v.
$$

这里 `zero` 是一个 FP16 affine offset，不要与某些整数 affine quantization
API 中的 integer zero-point 混淆。

### 7.4 一个简单例子

假设某个 V 向量：

```text
v_min = -1
v_max = 2
```

则：

$$
s_v=(2-(-1))/15=0.2,
$$

对于 (v=0.37)：

$$
q=\mathrm{round}((0.37+1)/0.2)=\mathrm{round}(6.85)=7,
$$

重建为：

$$
\hat v=7\times0.2-1=0.4.
$$

误差是 0.03。

### 7.5 V 最终保存什么？

对 `D=128,b=4`：

```text
V indices = 64 B
V scale   = FP16, 2 B
V zero    = FP16, 2 B
```

Scale 和 zero 都是 **每 token、每 KV head** 一份，不是全模型共享，也不是
每 coordinate 一份。

---

## 8. TurboQuant-MSE、TurboQuant-Prod 与 QJL

### 8.1 TurboQuant-MSE 优化什么？

TurboQuant-MSE 的目标是重建误差：

$$
\mathbb E\lVert x-\tilde x_{mse}\rVert_2^2.
$$

核心是：

```text
normalize -> random/structured orthogonal rotation
          -> Lloyd-Max scalar quantization
          -> scale/norm reconstruction
```

本项目 K 路径属于这个工程方向，并额外启用 norm correction。

### 8.2 为什么 MSE-optimal 不等于 inner-product optimal？

MSE quantizer 最小化向量重建误差，但 Attention 真正关心：

$$
q^Tk.
$$

即使 $\tilde k$ 的 MSE 很小，估计量 $q^T\tilde k$ 仍可能存在系统性 bias。
论文指出 MSE-optimal quantization 会导致 inner-product shrinkage/bias，因此
提出 TurboQuant-Prod。

### 8.3 TurboQuant-Prod 做什么？

对于总 bit budget $b$，论文的两阶段思路是：

1. 用 $b-1$ bit TurboQuant-MSE 得到 $\tilde x_{mse}$；
2. 计算 residual：

$$
r=x-\tilde x_{mse};
$$

3. 用额外 1 bit/coordinate 的 QJL 对 residual 编码；
4. 查询 inner product 时，将 MSE 部分与 QJL residual correction 相加。

目标不是更精确地逐坐标重建 $x$，而是构造低失真、无偏的 inner-product
estimator。

### 8.4 QJL 到底保存什么？

QJL 使用一个随机 projection matrix $S$，对 residual 做投影：

$$
u=Sr,
$$

然后只保存 sign：

$$
b_i=\mathrm{sign}(u_i)\in\{-1,+1\}.
$$

这就是 1 bit/coordinate。为了恢复 correction 的幅度，通常还需要 residual
norm 等少量 metadata，并在查询端对 q 做匹配的 projection/estimation。

因此 QJL 不是一个乘在 4-bit index 上的 scalar scale，而是一条独立的
random-projection residual channel。

### 8.5 QJL 与 K norm、V scale 的本质区别

| 量 | 信息类型 | 作用 |
| --- | --- | --- |
| K norm | 一个幅度标量 | 将单位方向恢复到原 K 大小 |
| V scale/zero | 两个 affine 参数 | 将 V integer level 映射回数值范围 |
| QJL signs | 每 coordinate 1 bit | 编码 MSE residual 的随机投影方向 |
| QJL residual norm | residual 幅度 metadata | 标定 QJL correction 大小 |

所以“QJL 是不是 scale”这个问题的答案明确是：**不是**。

### 8.6 0xSero/turboquant 使用了什么？

`0xSero/turboquant` 是一个独立开源实现。其文档描述了 random orthogonal
rotation、`b-1` bit Lloyd-Max MSE、1-bit QJL residual、V group quantization
和 bit packing，并包含 `TurboQuantMSE` 与 `TurboQuantProd`。

它可以帮助理解论文 Algorithm 1/2，但不能据此推断当前官方 vLLM backend
也保存 QJL payload。两个仓库的 cache format 和执行路径需要分别看源码。

### 8.7 当前 vLLM 为什么没有 QJL？

当前 vLLM `TurboQuantConfig` 文档明确写明 QJL 被有意省略，理由是多个社区
实现观察到 QJL 的 variance 经 softmax 放大后会损害 attention quality。

因此 vLLM 当前命名的 `turboquant_4bit_nc` 实际是：

```text
K: Hadamard + 4-bit Lloyd-Max MSE + norm correction
V: 4-bit uniform affine quantization
QJL: none
```

这是工程选择。论文中的 TurboQuant-Prod 理论仍然成立，但“论文定义了 QJL”
与“当前 vLLM production path 是否启用 QJL”是两个不同问题。

---

## 9. 当前 vLLM TurboQuant Backend 的完整流程

### 9.1 Backend 支持的模式

当前官方文档列出：

```text
turboquant_k8v4
turboquant_4bit_nc
turboquant_k3v4_nc
turboquant_3bit_nc
```

本项目固定研究：

```text
turboquant_4bit_nc
```

也就是 4-bit MSE K、4-bit uniform V 和 norm correction。

### 9.2 初始化阶段

Backend 按 head dimension 构造并缓存：

1. normalized Hadamard matrix $H$；
2. Lloyd-Max centroid table；
3. 相邻 centroid midpoint；
4. Decode 所需 workspace 和 metadata。

这些不是每个 token 重建一次。

### 9.3 Store 阶段

输入是新 token 的原始 K/V：

```text
K
 -> FP32 norm
 -> normalize
 -> Hadamard GEMM
 -> midpoint binary search
 -> 4-bit index packing
 -> norm correction
 -> K norm FP16 metadata

V
 -> per-token/head min/max
 -> scale/zero
 -> 4-bit uniform index
 -> bit packing
 -> scale/zero FP16 metadata
```

Store 根据 `slot_mapping` 写入 paged KV Cache。

### 9.4 Decode 阶段

Decode 输入是 rotated Q、packed cache、block table、sequence lengths、centroid
table 等。热路径为：

```text
block table mapping
 -> load packed K/V + metadata
 -> nibble unpack
 -> centroid lookup * K norm
 -> V index * scale + zero
 -> QK
 -> online softmax
 -> PV
 -> split partial output + LSE
```

Stage2 再对 split partials 做 log-sum-exp merge。

### 9.5 Prefill 与 Decode 为什么可能走不同路径？

Prefill 有大量 query token，适合标准 dense/FlashAttention 计算；Decode 通常
每请求一个 query，却读取长历史 cache，压缩 cache 的 bandwidth 收益更重要。
当前 vLLM backend 文档也区分 uncompressed prefill、compressed store 和
compressed decode；continuation prefill 可能先 dequant 历史 cache 再拼接当前
raw K/V。

---

## 10. 本项目 `turboquant_4bit_nc` Cache 布局

### 10.1 每个逻辑 slot

固定 `D=128`：

| 内容 | 大小 |
| --- | ---: |
| packed K indices | 64 B |
| packed V indices | 64 B |
| K corrected norm | 2 B |
| V scale | 2 B |
| V zero | 2 B |
| 合计 | 134 B |

原始 FP16 K+V payload 为：

$$
128\times2\text{ B}\times2=512\text{ B}.
$$

因此包含 metadata 后的 slot compression ratio 为：

$$
512/134\approx3.82\times.
$$

这与 vLLM preset 文档中 `turboquant_4bit_nc` 的约 3.8x 一致。

### 10.2 SoA physical block

一个 physical block 有 16 token、8 KV head：

```text
DATA REGION
[token][kv_head][K64 | V64]
= 16 * 8 * 128 B
= 16384 B

METADATA REGION
[kv_head][field][token]
field 0 = K norm
field 1 = V scale
field 2 = V zero
= 8 * 3 * 16 * 2 B
= 768 B

TOTAL = 17152 B
```

AoS 与 SoA 的逻辑信息相同，区别是 metadata 的物理排列。

### 10.3 “一个 FP16 norm”为什么不等于完整 scale 开销？

如果只讨论 K，确实可以概括为：

```text
每 coordinate 4-bit index + 每 vector 一个 FP16 norm
```

但整个 KV slot 还包含 V 的 FP16 scale/zero。Centroid table 虽然共享、摊销后
很小，也属于 Decode 必需数据。准确描述应始终说明讨论的是 K-only 还是 K+V。

---

## 11. 本项目 CUDA Decode 与算法的对应关系

### 11.1 CUDA Kernel 没有重新实现 Codebook 求解

Lloyd-Max centroid 在 Python/backend 初始化阶段生成。CUDA Stage1 接收已经
准备好的 16-entry FP32 centroid tensor，Kernel 只做 lookup，不在热路径求解
积分或更新 codebook。

### 11.2 Kernel 中的 K decode

本项目 V9 读取每个 packed K byte，拆成两个 nibble，查 centroid，并乘当前
token 的 FP16 corrected norm，然后以 FP16 tile 参与 `mma.sync.m16n8k16`。

### 11.3 Kernel 中的 V decode

V nibble 转换为 0..15 的 index，再执行：

```cpp
v = index * v_scale + v_zero;
```

重建的 FP16 V tile 直接参与 PV，不写回完整 global FP16 cache。

### 11.4 为什么融合 decode 很重要？

如果先生成完整 FP16 K/V：

```text
读 INT4 -> 写 FP16 -> 再读 FP16 -> Attention
```

会增加 global traffic 和 Kernel launch。本项目是：

```text
读 INT4 -> tile 内重建 -> 立即 QK/PV
```

因此量化节省的 memory bandwidth 能直接进入 Attention 热路径。

### 11.5 V1-V9 优化的是量化算法还是执行实现？

主要优化执行实现，没有改变 `turboquant_4bit_nc` 的 quantization semantics。
各版本读取同一逻辑 cache，输出相同 Stage1 语义。优化包括：

- GQA K/V 复用和 warp mapping；
- tiled online softmax；
- Tensor Core QK/PV；
- fixed workload specialization；
- fragment register direct writeback；
- packed `uint32` load 和 `half2` store；
- barrier fusion；
- GQA-4 `m16n8k16` mapping；
- FlashInfer-style register-resident softmax state。

算法质量由 rotation、codebook、bit width、norm correction、V quantization 等
决定；V1-V9 主要决定同一算法在 RTX 4090 上执行多快。

---

## 12. 常见误区逐条纠正

### 误区 1：旋转后每个 coordinate 均匀分布

错误。整个向量在球面上均匀；单 coordinate 是对称 Beta 型，高维近似
$\mathcal N(0,1/d)$。

### 误区 2：不同 coordinate 严格独立

错误。它们受单位范数约束。高维下有限数量 coordinate 近似 jointly independent
Gaussian，属于渐近结论。

### 误区 3：vLLM 从真实 KV 采样并训练 codebook

错误。本地实现直接对 $\mathcal N(0,1/d)$ PDF 做 Lloyd-Max 数值积分；没有
calibration dataset、sampling 或 K-means。

### 误区 4：Codebook 是一个大 vector codebook

错误。它只有 $2^b$ 个 scalar centroid，所有 coordinate 复用。4-bit 时只有
16 个 FP32 值。

### 误区 5：K 和 V 使用同一个 scale 和 quantizer

错误。K 使用 centroid index + norm；V 使用 uniform index + scale/zero。

### 误区 6：K norm 就是 Attention scale

错误。K norm 是 per-token/head metadata，Attention scale 是全局
$1/\sqrt d$。

### 误区 7：QJL 就是 residual scale

错误。QJL 是 residual random projection 的 1-bit sign channel；residual norm
只是它可能需要的 metadata 之一。

### 误区 8：TurboQuant-Prod 就是把 MSE scale 调一下

错误。TurboQuant-Prod 是 MSE quantizer 加 QJL residual correction 的两阶段
inner-product estimator。

### 误区 9：论文有 QJL，所以 vLLM cache 一定有 QJL bits

错误。当前 vLLM 明确省略 QJL，本项目的 134 B slot 也没有 QJL payload。

### 误区 10：Pure Hadamard 对任意输入都严格产生 Gaussian coordinate

错误。精确 Beta/Gaussian 推导对应随机球面 rotation；Hadamard 是结构化工程
替代，应靠实际分布和质量验证。

---

## 13. 一条完整的心智模型

可以把本项目理解成下面这条链：

```text
                         STORE

K_fp16[D]
  -> norm gamma
  -> normalize
  -> Hadamard rotation
  -> 16-entry Lloyd-Max midpoint bucketize
  -> 4-bit centroid indices
  -> corrected norm = gamma / ||centroid-vector||

V_fp16[D]
  -> min/max
  -> scale=(max-min)/15, zero=min
  -> 4-bit uniform indices

  -> packed SoA KV cache


                         DECODE

Q_fp16[D]
  -> matching Hadamard rotation

packed K
  -> nibble -> centroid -> * corrected norm
  -> QK / sqrt(D)

packed V
  -> nibble -> * scale + zero

QK scores
  -> online softmax
  -> probability * V
  -> split partial output + LSE
  -> Stage2 merge
```

其中不存在：

```text
QJL residual bits
QJL projection matrix
QJL correction term
```

这就是当前 vLLM `turboquant_4bit_nc` 与本项目 CUDA Decode 的准确边界。

---

## 14. 面试版简短回答

如果面试官问“TurboQuant 原理是什么”，可以回答：

> TurboQuant 先把每个 K 向量归一化，再做正交旋转。论文中随机旋转使单位
> 向量在球面上均匀，单 coordinate 呈对称 Beta 型分布，高维下近似
> $\mathcal N(0,1/d)$，有限坐标也近似独立，所以能用同一个 Lloyd-Max scalar
> quantizer 逐维编码。vLLM 不从真实 KV 训练 codebook，而是直接对
> $\mathcal N(0,1/d)$ 的 PDF 数值积分，生成 4-bit 时的 16 个 centroid。K cache
> 保存 centroid index 和一个 FP16 corrected norm；V 使用 per-token/head
> uniform INT4，保存 scale 和 zero。论文的 TurboQuant-Prod 还会对 MSE
> residual 做 1-bit QJL 来获得无偏内积估计，但 QJL 不是 scale，当前 vLLM
> 和我的 CUDA 项目都没有使用它。我的工作是在不改变这套量化语义的前提下，
> 将 unpack、反量化、Tensor Core QK/PV 和 online softmax 融合到 Decode
> Kernel 中。

---

## 15. 参考资料与对应源码

### 论文与独立实现

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
- [0xSero/turboquant](https://github.com/0xSero/turboquant)
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482)

### vLLM 官方文档

- [TurboQuant attention backend](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/turboquant_attn/)
- [TurboQuant quantization package](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/turboquant/)
- [Triton TurboQuant Decode](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/ops/triton_turboquant_decode/)
- [Attention backend feature support](https://docs.vllm.ai/en/stable/design/attention_backends/)

### 本仓库关键文件

- `reference/centroids.py`：Gaussian PDF 与 Lloyd-Max numerical integration；
- `reference/soa_store.py`：K normalize/rotate/bucketize/pack 与 V scale/zero；
- `reference/soa_decode_v1.py`：centroid/norm 和 V affine reconstruction；
- `reference/turboquant_attn.py`：Hadamard、centroid 初始化与 backend dispatch；
- `baseline/common.py`：本项目固定 shape 与 134 B SoA layout；
- `cuda/tq4_cuda_v9.cu`：融合 4-bit decode、MMA、online softmax 的最终 Stage1。

> 时间敏感说明：vLLM 文档与实现会继续变化。本文关于当前官方 backend 的描述
> 核对日期为 2026-08-25；论文算法与具体 vLLM 工程策略应始终分开理解。
