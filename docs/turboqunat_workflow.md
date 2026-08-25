# TurboQuant 核心原理与 vLLM 实现总结

这个问题非常关键，因为 **TurboQuant 不是简单的 INT4 KV Cache**。

你可以先建立这样一个总体认识：

> **TurboQuant 的核心思想，是先通过正交旋转把任意高维向量变换到一个具有稳定统计分布的空间，然后利用这个已知分布提前构造 Lloyd-Max 最优 scalar quantizer。运行时不需要针对模型训练 codebook，而只需要保存每个 coordinate 对应的低 bit index 和向量尺度信息。**

对于 vLLM 当前工程实现，还要额外记住：

> **K 和 V 的量化方法并不一样。K 使用 TurboQuant 的 rotation + centroid 路径；V 使用更普通的 affine/uniform quantization。**

我们一步一步看。

---

# 1. TurboQuant 为什么要做 Rotation？

假设一个 Key：

$$
K\in\mathbb{R}^{128}
$$

原始 K 可能类似：

```text
K =

[
0.02,
0.03,
3.87,
-0.01,
0.05,
...
]
```

存在：

```text
outlier
```

也就是说：

某几个 coordinate 特别大，而其他 coordinate 很小。

如果直接做低 bit scalar quantization：

```text
FP16 K

↓

INT4
```

就会出现一个问题：

为了覆盖那个大的 outlier，quantization range 必须扩大。

结果：

```text
大量普通值

↓

挤在少量量化级别附近

↓

量化误差增加
```

因此 TurboQuant 首先进行：

$$
y=\Pi x
$$

其中：

$$
\Pi^T\Pi=I
$$

也就是正交旋转。

---

# 2. Rotation 到底做了什么？

因为：

$$
\Pi^T\Pi=I
$$

所以：

$$
\lVert\Pi x\rVert_2=\lVert x\rVert_2
$$

也就是说：

> Rotation 不改变整个 vector 的 L2 norm。

但是它会重新混合每一个 coordinate。

原来：

```text
x =

[
3.8,
0.1,
0.1,
0.1,
...
]
```

经过 rotation 后可能变成：

```text
y =

[
0.21,
-0.13,
0.08,
0.17,
...
]
```

这里不要把它理解成：

> “每个 coordinate 变成 uniform distribution。”

这是不对的。

准确说法是：

> 如果 $x$ 是一个固定的单位向量，而 $\Pi$ 是随机正交旋转，那么整个旋转后的向量 $y=\Pi x$ 在单位球面上均匀分布。

也就是：

```text
整个vector的方向

↓

在sphere上uniform
```

但是：

```text
单独一个coordinate

≠

Uniform distribution
```

---

# 3. 那单个 coordinate 是什么分布？

假设：

$$
x\in S^{d-1}
$$

即：

$$
\lVert x\rVert_2=1
$$

随机旋转：

$$
y=\Pi x
$$

那么：

$$
y=[y_1,y_2,\dots,y_d]
$$

其中任意一个 coordinate：

$$
y_i
$$

的精确边缘分布是一个 **Beta 型分布**。

它的 PDF 为：

$$
f(x)=
\frac{\Gamma(d/2)}
{\sqrt{\pi}\Gamma((d-1)/2)}
(1-x^2)^{(d-3)/2}
$$

范围：

$$
x\in[-1,1]
$$

所以论文理论可以理解成：

```text
随机正交旋转

↓

整个vector在单位球面上均匀

↓

单个coordinate具有Beta型分布
```

---

# 4. 为什么又说高维情况下接近 Normal？

当：

$$
d
$$

非常大时，单位球面上的质量高度集中。

因此单个 coordinate：

$$
y_i
$$

会近似：

$$
y_i\sim \mathcal{N}(0,1/d)
$$

所以：

```text
论文精确理论：

Beta型分布

        ↓

高维极限

        ↓

Gaussian approximation

N(0,1/d)
```

例如：

```text
head_dim = 128
```

那么：

$$
\sigma^2=\frac1{128}=0.0078125
$$

标准差：

$$
\sigma=\frac1{\sqrt{128}}
\approx0.0884
$$

所以大部分 coordinate 都集中在：

```text
0附近
```

而不是到处出现大 outlier。

---

# 5. 为什么 coordinate 近似独立很重要？

这是 TurboQuant 很核心的一步。

原来的问题是：

```text
一个128维vector

↓

Vector Quantization
```

理论上各个维度之间可能存在复杂相关性。

但是经过随机 rotation：

```text
coordinate分布稳定

+

不同coordinate在高维下近似独立
```

那么我们就可以近似认为：

$$
y_0,y_1,\dots,y_{127}
$$

可以分别量化。

于是：

```text
复杂的高维Vector Quantization

↓

128个Scalar Quantization
```

但是这些 scalar quantizer：

**可以共享同一个 codebook。**

这就是 TurboQuant 为什么既有 Vector Quantization 的理论优势，又非常容易向量化实现。

---

# 6. 那 Codebook 到底从哪里来？

这是最重要的一点：

> **Codebook 不是从真实 KV Cache 数据训练出来的。**

不是：

```text
收集100万个Llama K

↓

训练K-means

↓

得到codebook
```

也不是：

```text
每个模型重新calibration
```

而是利用上面已经知道的：

$$
f(x)
$$

直接构造最优 scalar quantizer。

---

# 7. 假设做 4bit

4bit：

$$
2^4=16
$$

所以需要：

```text
16个 reconstruction value
```

即：

$$
c_0,c_1,\dots,c_{15}
$$

这些就是：

```text
centroids
```

或者：

```text
codebook
```

运行时，一个 coordinate 不再保存 FP16 value，而只保存：

```text
0 ~ 15
```

中的一个 index。

4 bit 刚好够：

```text
0000 ~ 1111
```

---

# 8. Lloyd-Max 优化目标是什么？

假设：

$$
X\sim f(x)
$$

我们希望找到：

$$
c_0,\dots,c_{15}
$$

让平均重建误差最小。

即：

$$
\min \mathbb{E}\left[(X-\hat{X})^2\right]
$$

如果划分成 16 个 quantization interval：

$$
R_0,R_1,\dots,R_{15}
$$

那么：

$$
D=\sum_i\int_{R_i}(x-c_i)^2f(x)\mathrm{d}x
$$

我们的目标就是：

$$
\min_{c_0,\dots,c_{15}}D
$$

本质：

> continuous 1-D k-means。

---

# 9. Lloyd-Max 第一步：初始化 centroid

假设为了理解，我们只有：

```text
4个centroid
```

最开始可以初始化：

```text
c0

c1

c2

c3
```

比如：

```text
-1.5
-0.5
+0.5
+1.5
```

这些只是：

```text
initial guess
```

不是最终 codebook。

---

# 10. Lloyd-Max 第二步：计算 boundary

假设：

```text
c0      c1      c2      c3
```

那么相邻 centroid 之间的 decision boundary：

$$
b_i=\frac{c_i+c_{i+1}}2
$$

例如：

$$
c_0=-1.5
$$

$$
c_1=-0.5
$$

那么：

$$
b_0=-1.0
$$

所以：

```text
      c0       boundary       c1

      -1.5       -1.0        -0.5
--------|----------|-----------|-------
```

如果：

$$
x<-1.0
$$

它更接近：

```text
c0
```

如果：

$$
x>-1.0
$$

它更接近：

```text
c1
```

---

# 11. Lloyd-Max 第三步：重新计算 centroid

这里是最重要的公式。

某一个区间：

$$
[a,b]
$$

新的 centroid 不是简单：

$$
\frac{a+b}{2}
$$

因为：

```text
区间里的概率密度不是uniform
```

正确的 centroid 是该区间内随机变量的条件期望：

$$
c=\mathbb{E}[X\mid a<X<b]
$$

也就是：

$$
c=\frac{\int_a^b x f(x)\mathrm{d}x}
        {\int_a^b f(x)\mathrm{d}x}
$$

分母：

$$
\int_a^b f(x)\mathrm{d}x
$$

表示：

这个区间里面有多少 probability mass。

分子：

$$
\int_a^b xf(x)\mathrm{d}x
$$

表示：

概率加权后的 value。

所以：

```text
boundary

↓

根据PDF重新算conditional mean

↓

new centroid
```

---

# 12. 为什么不断迭代？

因为：

```text
centroid决定boundary
```

但：

```text
boundary又决定centroid
```

所以：

```text
centroid

↓

boundary

↓

重新计算centroid

↓

重新计算boundary

↓

重新计算centroid

↓

...
```

直到：

$$
\left|c_i^{new}-c_i^{old}\right|
$$

非常小。

最终得到：

```text
固定centroid table
```

---

# 13. TurboQuant 论文和 vLLM 在这里稍微不同

论文理论使用：

```text
精确Beta型coordinate distribution
```

来描述问题。

但是 vLLM 的工程实现利用：

$$
d
$$

比较大时：

$$
Beta\approx Gaussian
$$

直接使用：

$$
X\sim\mathcal{N}(0,1/d)
$$

来构造 centroid。

所以：

```text
论文：

Sphere

↓

Beta

↓

Lloyd-Max
```

而 vLLM：

```text
Sphere

↓

高维Gaussian近似

↓

N(0,1/d)

↓

Lloyd-Max
```

---

# 14. vLLM 会不会从 Gaussian 里面采样数据？

**不会。**

这一点要记牢。

不是：

```text
N(0,1/d)

↓

随机sample 100万个数

↓

K-means
```

vLLM 直接知道 PDF：

$$
f(x)
$$

然后做：

```text
Gaussian PDF

↓

数值积分

↓

Lloyd-Max update
```

也就是直接算：

$$
\frac{\int_a^bxf(x)\mathrm{d}x}
     {\int_a^bf(x)\mathrm{d}x}
$$

因此：

> **没有真实 KV sampling，也没有 Monte Carlo sampling。**

---

# 15. 以 head_dim=128、4bit 为例

现在：

$$
d=128
$$

所以：

$$
\sigma=\frac1{\sqrt{128}}
\approx0.0884
$$

vLLM 使用：

$$
\mathcal{N}(0,0.0884^2)
$$

4bit：

```text
16 levels
```

然后 Lloyd-Max：

```text
初始化16个centroids

↓

计算15个boundaries

↓

对Gaussian PDF积分

↓

更新16个centroids

↓

重新计算boundaries

↓

...

↓

收敛
```

最终：

```text
centroid[16]
```

这张表之后可以重复使用。

---

# 16. 为什么 TurboQuant 可以固定 Codebook？

现在这个逻辑就完全连起来了：

普通 KV：

```text
不同模型

不同layer

不同token

不同coordinate

↓

原始分布很复杂
```

所以直接 scalar quantization：

可能需要 calibration。

TurboQuant：

```text
任意x

↓

normalize

↓

random rotation

↓

coordinate具有统一的理论分布
```

所以：

```text
不需要知道原始K到底是什么分布
```

只需要知道：

```text
dimension d
```

就可以构造：

$$
\mathcal{N}(0,1/d)
$$

对应的 codebook。

因此它是：

```text
data-oblivious
```

---

# 17. Runtime 真正量化一个 K 怎么做？

现在进入真实 KV Cache。

假设：

$$
K\in\mathbb{R}^{128}
$$

首先计算：

$$
\gamma=\lVert K\rVert_2
$$

例如：

```text
K

↓

sqrt(sum(K[i]^2))

↓

gamma
```

---

# 18. 为什么需要 norm？

TurboQuant 的 centroid 是针对：

$$
\lVert x\rVert_2=1
$$

的向量设计的。

但真实 K：

可能：

$$
\lVert K\rVert_2=7.2
$$

另一个：

$$
\lVert K\rVert_2=11.8
$$

所以先：

$$
\hat K=\frac{K}{\gamma}
$$

使：

$$
\lVert\hat K\rVert_2=1
$$

然后量化：

$$
\hat K
$$

同时单独保存：

$$
\gamma
$$

以后恢复幅度。

---

# 19. K 的完整量化过程

现在流程是：

```text
原始 FP16/BF16 K

        ↓

计算 L2 norm

γ = ||K||

        ↓

normalize

K_norm = K / γ

        ↓

orthogonal rotation

K_rot = Π K_norm

        ↓

每个coordinate

        ↓

寻找最近centroid

        ↓

保存index

        ↓

128 × 4bit

+

γ
```

也就是：

```text
K cache ≈

packed 4bit indices

+

norm
```

---

# 20. “查最近 centroid”是什么意思？

假设 codebook 中某几个值：

```text
c7 = ...
c8 = ...
c9 = ...
```

而：

$$
K_{rot}[0]
$$

介于：

```text
c8和c9
```

比较：

$$
\left|K_{rot}[0]-c_8\right|
$$

和：

$$
\left|K_{rot}[0]-c_9\right|
$$

谁小：

就保存谁的 index。

例如：

```text
nearest = c9

↓

index = 9

↓

1001
```

真正写入 KV Cache 的是：

```text
1001
```

不是 centroid float 本身。

---

# 21. Decode 时 K 怎么恢复？

最差的办法：

```text
INT4 KV

↓

dequant完整FP16 K

↓

写回显存

↓

重新读取FP16 K

↓

QK
```

这样非常浪费。

TurboQuant decode：

```text
load packed 4bit

↓

unpack index

↓

centroid[index]

↓

参与QK accumulation
```

不需要构造完整 FP16 K buffer。

这就是：

```text
on-the-fly dequantization
+
attention fusion
```

---

# 22. Query 为什么也必须 Rotation？

这是非常重要的数学关系。

原来：

$$
QK^T
$$

假设我们对 K 使用正交 rotation：

$$
K'=K\Pi^T
$$

或者按 column/vector convention 写成：

$$
K'=\Pi K
$$

那么 Query 也做对应变换。

因为正交矩阵满足：

$$
\Pi^T\Pi=I
$$

所以可以保持：

$$
QK^T
$$

不变。

直观理解：

```text
K换了一个坐标系

↓

Q也必须换到同一个坐标系

↓

才能继续做inner product
```

否则你是在：

```text
一个坐标系的Q

dot

另一个坐标系的K
```

结果当然不对。

---

# 23. 为什么 K 特别需要这种量化？

因为 K 直接决定：

$$
score_i=QK_i^T
$$

然后：

$$
P=\mathrm{softmax}(score)
$$

K 的误差：

$$
\Delta K
$$

会产生：

$$
\Delta score=Q\Delta K^T
$$

而 score 后面还要经过：

```text
softmax
```

因此误差可能改变 attention probability。

所以 K 的量化特别关心：

```text
geometry

inner product

direction
```

这正是 TurboQuant rotation + Lloyd-Max 的价值。

---

# 24. V 和 K 是同一种量化吗？

**不是。**

这一点现在必须明确记住。

K 路径：

```text
K

↓

norm

↓

rotation

↓

Lloyd-Max centroid index
```

Decode：

```text
index

↓

centroid[index]

↓

norm correction

↓

QK
```

---

V 在当前 vLLM 工程路径更接近：

```text
普通uniform / affine quantization
```

也就是：

```text
V

↓

min / max

↓

scale

↓

zero

↓

4bit value
```

恢复：

$$
V\approx q\times scale+zero
$$

不是：

```text
centroids[v_idx]
```

---

# 25. 为什么 V 不一定需要 centroid？

因为 V 的作用不同。

K：

$$
QK^T
$$

决定：

```text
attention probability
```

V：

$$
PV
$$

本质：

$$
O=\sum_iP_iV_i
$$

是 weighted sum。

所以工程上可以针对两者采用不同量化策略：

```text
K:

更重视inner-product preservation


V:

更适合简单、快速的affine dequant
```

因此：

> **不能再说 K 和 V 都查 centroid。**

---

# 26. Scale / Norm 到底是什么？

这个非常容易和 QJL 混淆。

对于 K：

```text
norm
```

主要作用：

> 恢复原向量幅度。

例如：

$$
K=\gamma\hat K
$$

其中：

$$
\gamma=\lVert K\rVert_2
$$

所以 quantizer 主要处理：

$$
\hat K
$$

而：

$$
\gamma
$$

单独保存。

---

对于 V：

```text
scale
+
zero
```

是普通 affine quantization 参数。

例如：

$$
V\approx q\cdot scale+zero
$$

---

# 27. QJL 是 scale 吗？

**绝对不是。**

这是必须彻底区分的。

Scale / norm：

```text
恢复vector幅度
```

QJL：

```text
补偿第一阶段quantization产生的residual
```

两者完全不同。

---

# 28. TurboQuant_mse 是什么？

我们前面讨论的：

```text
rotation

↓

Lloyd-Max

↓

centroid index
```

实际上主要对应：

```text
TurboQuant_mse
```

目标：

$$
\min \mathbb{E}\left[\lVert x-\hat{x}\rVert_2^2\right]
$$

也就是：

> 让 reconstruction MSE 尽可能小。

---

# 29. 为什么还需要 TurboQuant_prod？

问题在于：

MSE 小不代表：

$$
Q\hat K
$$

是无偏的。

真正 attention 关心：

$$
QK^T
$$

论文发现：

MSE-optimal quantizer 的 inner product estimator 会存在 bias。

所以提出：

```text
TurboQuant_prod
```

目标：

让：

$$
\mathbb{E}\left[\langle y,\hat{x}\rangle\right]
=\langle y,x\rangle
$$

尽可能成立。

---

# 30. TurboQuant_prod 怎么做？

假设总 bit budget：

```text
b bit
```

第一阶段：

```text
b-1 bit
TurboQuant_mse
```

得到：

$$
\hat x_{mse}
$$

然后计算：

$$
r=x-\hat x_{mse}
$$

这个：

```text
r
```

就是 residual。

然后再对 residual 使用：

```text
1-bit QJL
```

所以：

```text
x

↓

MSE quantization

↓

x_hat

↓

计算 residual

r=x-x_hat

↓

QJL(r)
```

---

# 31. QJL 到底做什么？

QJL：

Quantized Johnson-Lindenstrauss。

它大致做：

$$
Sr
$$

然后：

$$
\mathrm{QJL}(r)=\mathrm{sign}(Sr)
$$

也就是说：

只保存：

```text
+1 / -1
```

即：

```text
1 bit per coordinate
```

同时还需要 residual magnitude 等信息。

所以 TurboQuant_prod 保存的信息更像：

```text
MSE indices

+

QJL residual bits

+

residual norm / gamma
```

---

# 32. 为什么 vLLM 当前没有使用 QJL？

这是：

```text
论文最优性

vs

GPU工程效率
```

之间的 trade-off。

如果加入 QJL：

decode 还需要：

```text
load QJL bits

↓

unpack

↓

residual reconstruction

↓

额外random projection相关计算

↓

加入QK
```

意味着：

* 更多 KV Cache bytes
* 更多 load
* 更多 integer instruction
* 更多 arithmetic
* 更复杂 kernel

而 decode 最大目标是：

```text
减少HBM traffic
+
减少latency
```

所以当前 vLLM 工程路径采用了更简单的方案。

可以记：

```text
论文：

TurboQuant_mse
+
TurboQuant_prod(QJL)


vLLM工程：

主要使用MSE/centroid这一侧

没有完整QJL residual path
```

---

# 33. 所以你的 CUDA 项目本质上在做什么？

你的：

```text
turboquant-cuda-decode
```

主要不是研究：

```text
怎么训练TurboQuant
```

而是在研究：

> **如何把 vLLM 中已经压缩好的 TurboQuant KV Cache，在 decode 阶段以更高效率直接完成 attention。**

也就是：

```text
compressed KV in HBM

        ↓

CUDA load

        ↓

bit unpack

        ↓

K centroid decode
V affine decode

        ↓

QK

        ↓

online softmax

        ↓

PV

        ↓

output
```

---

# 34. History=4096 时 TurboQuant 在做什么？

假设现在：

```text
history = 4096
```

生成当前 token。

Query：

```text
Q_new
```

需要和：

```text
K1
K2
...
K4096
K_current
```

做 attention。

所以每生成一个 token：

```text
Q current

↓

scan大量历史compressed K

↓

QK score

↓

softmax

↓

scan compressed V

↓

PV
```

context 越长：

```text
读取KV Cache的数据越多
```

因此 TurboQuant 节省 KV bytes 的价值越来越明显。

---

# 35. 为什么 TurboQuant 不一定带来严格 4× Speedup？

虽然：

```text
FP16 = 16 bit

INT4 = 4 bit
```

理论存储缩小：

```text
4×
```

但是 kernel 还增加了：

```text
bit unpack

+

centroid lookup

+

scale/norm处理

+

integer instruction

+

address calculation
```

所以时间：

$$
T=
T_{memory}
+
T_{decode}
+
T_{attention}
$$

memory 部分可能降低接近 4×：

但是整体 latency：

不会严格 4×。

---

# 36. 为什么仍然值得做？

因为原始 decode attention：

通常：

```text
Arithmetic intensity很低
```

瓶颈主要：

```text
HBM bandwidth
```

TurboQuant 相当于：

```text
少搬很多byte

↓

多做一些便宜的integer / lookup / FMA
```

也就是典型：

> **用计算换带宽。**

GPU 的 compute 能力通常远高于 decode 场景实际需要的计算能力，所以这个 trade-off 是合理的。

---

# 37. 最后把整个 TurboQuant 流程串起来

## 第一部分：Codebook 构造

```text
dimension d

↓

论文推导：

rotation后coordinate具有Beta型分布

↓

高维近似：

N(0,1/d)

↓

vLLM直接使用Gaussian PDF

↓

初始化2^b个centroids

↓

根据centroid计算boundary

↓

对Gaussian PDF数值积分

↓

更新conditional mean

↓

重复Lloyd-Max

↓

得到固定centroid table
```

这个过程：

```text
不依赖真实KV Cache

不需要模型calibration

不在每次推理时运行
```

---

# 38. 第二部分：K 写入 KV Cache

```text
FP16/BF16 K

↓

计算norm

γ=||K||

↓

normalize

K/γ

↓

rotation

↓

每个coordinate

↓

nearest Lloyd-Max centroid

↓

保存4bit index

+

保存norm
```

---

# 39. 第三部分：V 写入 KV Cache

```text
FP16/BF16 V

↓

计算quantization parameters

↓

scale
+
zero

↓

uniform / affine quantization

↓

4bit V
```

注意：

```text
V不是K的centroid查表路径
```

---

# 40. 第四部分：Decode

Query：

```text
Q

↓

对应rotation

↓

Q_rot
```

K：

```text
packed K indices

↓

4bit unpack

↓

centroid[index]

↓

norm correction

↓

和Q_rot做dot
```

得到：

```text
attention score
```

然后：

```text
scores

↓

softmax
```

V：

```text
packed V indices

↓

4bit unpack

↓

q * scale + zero

↓

乘softmax probability

↓

accumulate
```

最后：

```text
Attention Output
```

---

# 41. 一张最重要的总图

```text
                   TurboQuant
                       |
                       |
          +------------+-------------+
          |                          |
          |                          |
       理论层                       工程层
          |                          |
          v                          v

 Random Rotation                  vLLM
          |                          |
          v                          |
 coordinate Beta                    |
          |                          |
          v                          |
 High-D Gaussian                     |
 N(0,1/d)                            |
          |                          |
          v                          |
 Lloyd-Max                           |
          |                          |
          v                          |
 fixed centroids --------------------+
                                     |
                       +-------------+-------------+
                       |                           |
                       v                           v
                       K                           V
                       |                           |
                    norm                      scale + zero
                       |                           |
                   rotation                      INT4
                       |
              Lloyd-Max index
                       |
                     INT4


                     Decode
                       |
             +---------+---------+
             |                   |
             K                   V
             |                   |
          unpack              unpack
             |                   |
        centroid lookup      affine dequant
             |                   |
          QK dot             softmax * V
             |                   |
             +---------+---------+
                       |
                     output
```

---

# 42. 最后一段面试回答

如果面试官问：

> TurboQuant 的核心原理是什么？

你可以回答：

> TurboQuant 的核心思想是先对高维向量进行随机正交旋转。对于归一化向量，旋转后的整个向量均匀分布在单位球面，而单个 coordinate 具有 Beta 型边缘分布，在高维情况下可以近似为 $\mathcal{N}(0,1/d)$，并且不同 coordinate 近似独立。因此一个复杂的高维 vector quantization 问题就可以近似转化成共享 codebook 的 scalar quantization。
>
> Codebook 不需要从真实模型的 KV Cache 中训练。TurboQuant 根据 rotation 后 coordinate 的理论分布，通过 Lloyd-Max optimization 离线求解最优 centroid。vLLM 的实现进一步使用 $\mathcal{N}(0,1/d)$ 的高斯近似，不采样真实 KV，也不做 Monte Carlo，而是直接对 Gaussian PDF 数值积分，迭代求出 centroid。
>
> 对 Key，运行时先保存 L2 norm，将 K normalize 后进行 rotation，再把每个 coordinate 映射到最近的 Lloyd-Max centroid，只保存低 bit index 和 norm。Decode 时 Query 也变换到相同的旋转空间，然后直接从 compressed K 中 unpack index、查 centroid 并参与 QK accumulation，不需要恢复完整 FP16 K。
>
> vLLM 中 V 与 K 并不是同一种量化路径，V 更接近普通的 affine quantization，通过 scale 和 zero 恢复。论文另外提出 TurboQuant_prod，用一阶段 MSE quantization 加 residual QJL 来获得无偏的 inner-product estimator；QJL 是 residual compensation，不是 scale。当前 vLLM 的 TurboQuant decode 工程路径并没有完整使用这套 QJL residual 方案。

这套说法基本就把 **论文理论、Lloyd-Max、vLLM 和你后面的 CUDA kernel** 全部串起来了。
