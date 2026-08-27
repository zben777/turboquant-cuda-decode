
# Quantization：量化基础 + GPTQ / AWQ / SmoothQuant / KIVI / KVQuant（复习版本）

---

## 一、量化基础

### 1. 先从 Linear 的计算开始

大模型中大量计算本质上都是线性层。

为了方便理解，可以统一写成：

$$
Y = XW
$$

假设：

$$
X: [M, K]
$$

$$
W: [K, N]
$$

那么：

$$
Y: [M, N]
$$

其中：

- $M$：token 数、batch×token 等；
- $K$：input channel / input feature；
- $N$：output channel / output feature。

所以：

```text
X: [M, K]
       ↑
   input channel

W: [K, N]
    ↑  ↑
 input output

Y: [M, N]
       ↑
   output channel
```

如果采用这个数学 layout：$W: [K, N]$，那么：

**W 的一列，对应一个 output channel。**

例如：

```text
                    output channel
                 out0  out1  out2

input c0          w00   w01   w02
input c1          w10   w11   w12
input c2          w20   w21   w22
input c3          w30   w31   w32
```

其中：

$$
[w_{00}, w_{10}, w_{20}, w_{30}]
$$

共同产生 $Y[..., \text{out}_0]$。

### 2. PyTorch 中为什么经常看起来反过来？

PyTorch `nn.Linear` 的 Weight 通常实际存储成：

$$
W: [N, K]
$$

也就是：

$$
W[\text{out\_channel}][\text{input\_channel}]
$$

所以计算通常等价写成：

$$
Y = XW^T
$$

此时：

```text
                input channel
             c0   c1   c2   c3

out0        [w00  w01  w02  w03]
out1        [w10  w11  w12  w13]
out2        [w20  w21  w22  w23]
 ↑
output channel
```

因此：

**PyTorch layout 中，W 的一行对应一个 output channel。**

所以以后不能死记"per-channel = 按行"或者"per-channel = 按列"，而应该先问：**哪个 Tensor？哪个 channel？当前矩阵 layout 是什么？**

### 3. 权重量化的基本目的

原始：

$$
Y = XW
$$

Weight 量化之后：

$$
W \rightarrow \hat{W}
$$

于是：

$$
Y_q = X\hat{W}
$$

我们真正希望的是：

$$
\boxed{X\hat{W} \approx XW}
$$

所以量化并不是要求每一个 Weight 都尽可能和原来的数值完全一样。

更根本的目标是：

**在降低模型存储、显存带宽和计算成本的同时，让最终模型输出尽可能接近全精度模型。**

这也是 GPTQ、AWQ 等算法为什么会利用 Activation 信息。虽然最后量化的是 Weight，但真正关心的是 $XW$ 受到多大影响。

### 4. Scale 和 Zero-point

把高精度浮点数转换成低 bit 整数时，需要建立浮点数和整数之间的映射。

以 asymmetric quantization 为例：

$$
q = \text{round}(x/s) + z
$$

其中：

- $s$：scale；
- $z$：zero-point；
- $q$：INT8 / INT4 / INT2 等整数。

反量化：

$$
\hat{x} = (q - z)s
$$

所以低 bit 量化通常不只是存 INT4 / INT2 data，还需要相应的 **scale** 和 **zero-point**。

有些实现不直接存整数 zero-point，而会存 scale + min 等等，但本质仍然是在保存量化映射所需要的参数。

### 5. 量化粒度

量化粒度本质上就是：**哪些元素共享同一套量化参数。**

**Per-tensor**

整个 Tensor 共用：

- 一个 scale
- 一个 zero

优点：metadata 少、实现简单。

缺点：一个 outlier 可能影响整个 Tensor。

**Per-channel**

每个 channel 一套参数。

对于 Weight 来说，常见 Weight per-channel quantization 是**每个 output channel 独立一套量化参数**。

PyTorch layout：$W[\text{out\_channel}][\text{input\_channel}]$

```text
out0: [ .... ] → scale0
out1: [ .... ] → scale1
out2: [ .... ] → scale2
```

**Per-group**

把一个 channel 内部再切成更小的 group。

例如，一个 output channel：

```text
[w0 ... w127]   → scale0
[w128 ... w255] → scale1
[w256 ... w383] → scale2
```

如果 `group_size = 128`，就是每 128 个 Weight 共用一套 qparams。

因此相比 per-channel：per-group 的粒度更细，outlier 的影响范围也更小。但代价是 scale / zero metadata 增多。

**Per-token**

主要常见于 Activation / KV Cache。

```text
token0: [x0, x1, ...] → qparams0
token1: [x0, x1, ...] → qparams1
token2: [x0, x1, ...] → qparams2
```

也就是每个 token 根据自己的数值分布进行量化。

### 6. 粒度越细意味着什么？

可以简单记：

> **粒度越细，outlier 的污染范围越小，通常精度越好；但 qparams metadata 和计算开销也会增加。**

### 7. W4A16 / W8A8

**W4A16**

- Weight → 4 bit
- Activation → 16 bit

典型：GPTQ、AWQ，属于典型 Weight-only quantization。

**W8A8**

- Weight → INT8
- Activation → INT8

典型：SmoothQuant。这时候除了减少 Weight 存储之外，还可以直接利用 INT8 GEMM / Tensor Core。

需要注意：W4A16 / W8A8 描述的是 Weight 和 Activation 的精度，**并不意味着 accumulator 一定也是对应精度**。

---

## 二、GPTQ

### 1. GPTQ 是什么？

GPTQ 是一种：

- **PTQ**（Post-Training Quantization）
- **Weight-only quantization**

典型：W4A16。

通常还会使用 **per-group quantization**，例如 `group_size = 128`。

### 2. GPTQ 想解决什么问题？

如果直接 RTN（Round-To-Nearest）：

$$
W \rightarrow \hat{W}
$$

那么：

$$
X\hat{W} - XW
$$

可能比较大。

GPTQ 的核心想法就是：

> **既然一个 Weight 量化以后产生了误差，那么能不能修改其他尚未量化的 Weight，让最终输出误差变小？**

答案就是 GPTQ 的**误差补偿**。

### 3. Calibration data 是干什么的？

GPTQ 会拿少量 calibration data 跑模型，获得当前 Linear 层的输入 Activation $X$，然后根据这些 Activation 构造二阶信息。

直观上可以理解：

$$
H \propto X^T X
$$

Hessian 告诉 GPTQ：

- 哪些 input feature 对输出比较敏感；
- 不同 input feature 之间有什么相关性；
- 一个 Weight 被量化以后，应该让哪些其他 Weight 帮忙吸收误差。

所以虽然 GPTQ 最终只量化 Weight，它仍然需要 Activation 来判断 Weight 的量化误差对输出有多大影响。

### 4. GPTQ 中"通道"到底是什么？

仍然看：

$$
Y = XW
$$

其中 $X: [M, K]$，所以 Hessian：

$$
X^T X: [K, K]
$$

这里的 $K$ 就是：

$$
\boxed{\text{input channel}}
$$

因此 GPTQ 的 Hessian 描述的是 **input-channel / input-feature 之间的二阶关系**。

如果用 PyTorch Weight layout：

```text
                 input channel
              c0   c1   c2   c3

out0         [w00  w01  w02  w03]
out1         [w10  w11  w12  w13]
out2         [w20  w21  w22  w23]
```

GPTQ 可以理解为**沿 input-channel 方向逐步处理**。例如：

1. 先处理 c0，c0 对应 Weight 产生量化误差；
2. 根据 Hessian 更新后面的 c1, c2, c3；
3. 再处理已经修改后的 c1，更新 c2, c3；
4. 再处理 c2……

与此同时，不同 output channel 对应的 Weight row 可以并行执行相关操作。

### 5. GPTQ 的核心流程

可以记成：

```text
Calibration dataset
        ↓
获得 Activation X
        ↓
构造 Hessian / 二阶信息
        ↓
开始量化 Weight
        ↓
量化当前部分
        ↓
得到 quantization error
        ↓
根据 Hessian 修改后续尚未量化的 Weight
        ↓
再量化修改后的 Weight
        ↓
不断继续
```

所以不能简单说"误差累计以后传给下一个 Weight"。

更准确是：

> **量化当前 Weight → 产生误差 → 根据 Hessian 更新后续未量化 Weight → 下一步再量化已经被更新后的 Weight。**

### 6. Per-group 在哪里？

最终 Weight 通常还是使用 **INT4 per-group**。

例如，一个 output channel：

```text
input c0 ~ c127   → 一套 scale / zero
input c128 ~ c255 → 另一套 scale / zero
```

这里的 group 是 **Weight quantization qparams 的粒度**。

要特别区分：

- quantization `group_size`
- GPTQ 算法内部为了提高计算效率使用的 processing block / block size

**这不是一个概念。**

### 7. GPTQ 最终记忆

> GPTQ 是 Weight-only PTQ。它利用 calibration Activation 构造 input-channel 维度上的 Hessian 信息，在逐步量化 Weight 的过程中，用后续尚未量化的 Weight 来补偿当前产生的量化误差，从而让 $X\hat{W}$ 尽可能接近 $XW$。最终 Weight 通常以 W4 per-group 的方式存储。

一句话：

$$
\boxed{\text{GPTQ} = \text{Hessian} + \text{量化误差补偿}}
$$

---

## 三、AWQ

### 1. AWQ 是什么？

AWQ：**Activation-aware Weight Quantization**。

典型：W4A16。

也是：

- PTQ
- Weight-only quantization

### 2. AWQ 的核心观察

AWQ 发现：不是所有 Weight 对模型输出都同样重要。

但是问题来了：怎么判断某个 Weight 是否重要？

AWQ 发现**不能简单只看 Weight 自己数值大不大**，而应该看它对应的 **Activation**。

### 3. 为什么看 Activation？

对于：

$$
Y = XW
$$

可以直观地理解成不同 input channel 的贡献相加：

$$
Y = \sum_j X_j W_j
$$

如果 $X$ 的 c1 经常很大，那么 $X_{c1} W_{c1}$ 这一项对最终 $Y$ 的影响可能比较大。

所以：**和这个 input channel 相连接的 Weight 更值得保护。**

### 4. AWQ 中的 channel 到底是什么？

假设 PyTorch layout：

```text
                 input channel
              c0   c1   c2   c3

out0         [w00  w01  w02  w03]
out1         [w10  w11  w12  w13]
out2         [w20  w21  w22  w23]
```

Calibration 发现 Activation 的 c1 特别重要，那么 AWQ 关注的是：

$$
w_{01},\ w_{11},\ w_{21},\ \dots
$$

也就是**连接到 input channel c1 的这一整条 Weight**。

所以 AWQ 的 activation-aware scaling 是：

$$
\boxed{\text{input channel-wise}}
$$

而不是只保护某一个单独 Weight。

### 5. AWQ 怎么保护这些 Weight？

AWQ 使用 scaling。可以理解为：

$$
X'_j = X_j / s_j
$$

$$
W'_j = s_j W_j
$$

于是：

$$
X'W' = XW
$$

所以在没有量化的时候，数学上仍然等价。

如果某个 input channel 很重要，对应 Weight × 一个合适的 $s$，就可以让这些 Weight 在后续低 bit quantization 中获得更好的相对表示精度。

### 6. s 怎么来？

AWQ 会通过 calibration Activation 得到每个 input channel 的统计信息，然后根据类似：

$$
s_j = (s_{X,j})^\alpha
$$

构造 scaling。

其中 $\alpha$ 不是固定拍脑袋得到，而是在候选范围里**搜索**，使量化后的输出误差比较小。

所以：

```text
Calibration
    ↓
得到 activation statistics
    ↓
判断哪些 input channels 重要
    ↓
搜索 scaling
    ↓
保护相应 Weight
```

### 7. AWQ 的一个常见误区

不要简单记成"找出最重要的 1% Weight，1% 用 FP16，99% 用 INT4"。

论文中的 salient weights 现象说明：少量 Weight 对精度非常关键。但是如果真的使用非常不规则的 FP16 sparse Weight + INT4 dense Weight，**硬件并不友好**。

所以最终 AWQ 的关键技巧是：

> **不需要真的把重要 Weight 全部留下 FP16，而是通过 scaling 去保护它们。**

然后 Weight 仍然可以保持规则的 INT4 per-group 存储。

### 8. AWQ 中两种"粒度"一定要分清

**AWQ scaling**：沿 input channel。

例如 Activation c1 + 所有连接 c1 的 Weight 共用相关的 scaling 逻辑。

**最终 Weight quantization**：通常每个 output channel 内沿 input-channel 方向再按照 `group_size` 切 group。

例如：

```text
out0:
  input 0~127   → 一套 INT4 qparams
  input 128~255 → 另一套 qparams
```

所以：**AWQ 的 scaling channel 和最终 Weight quantization group 不是一回事。**

### 9. AWQ 最终记忆

> AWQ 虽然只量化 Weight，但利用 calibration Activation 判断哪些 input channels 对输出更重要，并对连接这些 input channels 的 Weight 做 scaling 保护，然后再做 W4A16 per-group Weight quantization。

一句话：

$$
\boxed{\text{AWQ} = \text{Activation 告诉我哪些 Weight 重要} + \text{scaling 保护它们}}
$$

---

## 四、SmoothQuant

### 1. SmoothQuant 是什么？

SmoothQuant 主要用于 **W8A8**：

- Weight → INT8
- Activation → INT8

它最主要解决：**Activation 比 Weight 更难量化**，尤其是 Activation outlier。

### 2. 为什么 Activation 很难量化？

例如：

```text
             c0    c1    c2    c3

token0      2.0   0.5   0.8   0.3
token1      0.4   7.0   0.2   0.6
token2      0.5   6.5   0.4   0.2
```

可以看到 c1 长期存在比较大的数值。

如果整个 Activation 直接做低精度量化，7.0 / 6.5 会把量化 range 拉得很大，结果 0.2 / 0.3 / 0.4 这些正常的小值获得的有效精度就下降。

### 3. SmoothQuant 的核心思想

既然 Activation 很难量化、Weight 相对更容易量化，那么：

> **把一部分 Activation 的量化难度迁移给 Weight。**

原始：

$$
Y = XW
$$

针对 input channel $j$：

$$
X'_j = \frac{X_j}{s_j}
$$

同时：

$$
W'_j = s_j W_j
$$

于是：

$$
X'W' = XW
$$

数学结果不变。

### 4. SmoothQuant 中的 channel 是哪个 channel？

这里非常重要。SmoothQuant 的 smoothing scale $s_j$ 是：

$$
\boxed{\text{input channel-wise}}
$$

例如：

```text
             input channel
             c0   c1   c2   c3

token0      [..   ..   ..   ..]
token1      [..   7.0  ..   ..]
                  ↑
               outlier
```

假设 c1 对应 $s_1$，那么所有 token 的 $X[c1]$ 除以 $s_1$。

与此同时 Weight：

```text
                 input channel
              c0   c1   c2   c3

out0         [..   w01  ..   ..]
out1         [..   w11  ..   ..]
out2         [..   w21  ..   ..]
                   ↑
这一整条 input channel：
w01, w11, w21, ...
```

全部乘 $s_1$，这样才能保证 $X'W' = XW$。

### 5. SmoothQuant 的 scale 怎么得到？

经典 SmoothQuant 会利用 calibration 数据获得 Activation 和 Weight 的 channel-wise 最大幅值。

其核心 scaling 可以写成：

$$
s_j = \frac{\max|X_j|^\alpha}{\max|W_j|^{1-\alpha}}
$$

其中：

- $\max|X_j|$：第 $j$ 个 input channel 的 Activation 统计；
- $\max|W_j|$：Weight 对应 input channel 的幅值统计；
- $\alpha$：控制量化难度在 X 和 W 之间如何分配。

核心不需要死背公式，重点是：

> **$s$ 同时考虑 Activation 和 Weight 的分布，然后决定把多少量化难度从 X 转移到 W。**

### 6. SmoothQuant 离线做什么？

```text
Calibration dataset
       ↓
跑模型
       ↓
收集各层 Activation statistics
       ↓
结合 Weight statistics
       ↓
计算每个 input channel 的 s
       ↓
对 Weight 做对应 scaling
```

这部分在部署前完成。

### 7. 推理时做什么？

概念上：

```text
X
↓
按照对应 input-channel scaling
↓
Activation quantization
↓
INT8

Weight
↓
已经提前做过 smoothing
↓
INT8

最后：INT8 GEMM
```

实际部署时，这些 scaling 往往还可以进一步**融合到相邻算子或权重参数中**，避免真的单独启动一个"除以 s"的 kernel。

### 8. SmoothQuant 中又有两种"channel"

一定要区分：

**SmoothQuant smoothing**：input channel-wise，目的是 $X/s$、$W \times s$。

**Weight 最终 INT8 quantization**：Weight 自己的量化粒度可以采用 per-channel，这个 Weight per-channel 通常指 **output channel**。

所以这是**两个完全不同的"per-channel"**。

### 9. SmoothQuant vs AWQ

两者都可能出现 $X/s$、$W \times s$，但是目的不一样。

| | AWQ | SmoothQuant |
|---|---|---|
| 动机 | Activation 告诉我哪些 Weight 重要 | Activation outlier 太难量化 |
| 动作 | 保护重要 Weight | 把一部分难度迁移给 Weight |
| 结果 | 主要做 W4A16 | 让 Activation 也能 INT8 → W8A8 |

所以：

> **AWQ 是"保护 Weight"；SmoothQuant 是"平衡 Weight 和 Activation 的量化难度"。**

### 10. SmoothQuant 最终记忆

> SmoothQuant 沿 input channel 做 $X/s$ 和 $W \times s$，在保持 $Y=XW$ 数学等价的前提下，把 Activation outlier 的量化难度迁移一部分给 Weight，从而让 Weight 和 Activation 都可以更好地做 INT8，即 W8A8。

一句话：

$$
\boxed{\text{SmoothQuant} = \text{Activation 太难量化} \rightarrow \text{把一部分难度迁移给 Weight}}
$$

---

## 五、KIVI

### 1. KIVI 量化的已经不是 Weight

GPTQ / AWQ / SmoothQuant 主要讨论 Linear 的 Weight 和 Activation。

KIVI 开始讨论 **KV Cache**。

Decoder 生成过程中：

```text
token0 → K0, V0
token1 → K1, V1
token2 → K2, V2
...
```

历史 K/V 会不断增加。因此长 context 时，KV Cache 会消耗大量显存和显存带宽。KIVI 就是为了压缩它。

### 2. 一个 KV head 的结构

假设 `head_dim = D`，sequence length = T，那么一个 head 的 Key：

$$
K: [T, D]
$$

Value：

$$
V: [T, D]
$$

可以画成：

```text
                       KV channel / head_dim
                  c0    c1    c2   ...   c127

token0            k     k     k           k
token1            k     k     k           k
token2            k     k     k           k
...
```

这里，KIVI 中说的 **Key channel，就是一个 KV head 的 head_dim 中的某一个维度**。它和 Linear 的 output channel 已经完全不是一回事。

### 3. KIVI 最重要的观察

KIVI 发现：

$$
\boxed{K \rightarrow \text{per-channel}}
$$

而：

$$
\boxed{V \rightarrow \text{per-token}}
$$

原因可以简单理解为：Key 的 outlier 更表现出 channel-wise 的结构，而 Value 更适合按照每个 token 自己的 vector 分布进行量化。

### 4. K 的 per-channel 到底怎么量化？

这是 KIVI 最重要的地方。

假设 `group_size = 32`，沿 token / sequence 方向看：

```text
token 0 ~ 31  → token group 0
token 32 ~ 63 → token group 1
token 64 ~ 95 → token group 2
```

对于 group0：

```text
                  K channel
             c0    c1   ...   c127

token0       ...
token1       ...
...
token31      ...
```

然后：

- group0 的 c0：32 个值 → 一套 qparams
- group0 的 c1：32 个值 → 一套 qparams
- …
- group0 的 c127：32 个值 → 一套 qparams

所以 K 的量化参数索引，本质可以理解成：

$$
\boxed{\text{token group} \times \text{K channel}}
$$

### 5. 下一个 group 怎么办？

group1（token32 ~ token63）重新根据这一组的数据统计。

所以：

```text
group0, c0 → s00, z00
group1, c0 → s01, z01
group2, c0 → s02, z02
```

虽然都是 K channel c0，但 token group 不一样，因此 **qparams 通常也不一样**。这就是我们之前反复确认的重点。

### 6. V 为什么是 per-token？

V 的形状：

$$
V: [T, D]
$$

当一个新 token 产生：

$$
V_{\text{new}} = [v_0, v_1, \dots, v_{127}]
$$

这个 token 的整个 Value vector 已经完整了，所以可以直接：

```text
V_token0 → 自己量化
V_token1 → 自己量化
V_token2 → 自己量化
```

也就是 per-token。

如果具体实现还在 head_dim 内按照 group_size 再细分，那么一个 token 可以有多组 qparams。但是大的方向仍然是：

> **V 沿 token 方向独立量化，而不是像 K 那样固定某个 channel 再沿 sequence 收集数据。**

### 7. 为什么需要 recent / residual FP16？

K 有一个问题：它需要同一个 K channel + 多个 token 才能形成一个量化 group。

例如第一个新 token 刚来，$K_{\text{new}}[c0]$ 只有一个值。现在还没有后面的 token，因此没办法完成一个完整 token group 的 channel-wise quantization。

所以 KIVI 会保留一部分最新 KV（recent / residual KV）保持 **FP16**，旧的数据再进入 quantized cache。

### 8. residual_length 和 group_size 不一样

这一点我们专门确认过：

- `group_size`：一次 quantization group 包含多少个数值。对于 K 来说，可以体现为沿 token 方向多少个值形成一组。
- `residual_length`：最新多少 token 暂时保留全精度。

它们在某个配置里可能 `group_size = 32`、`residual_length = 32`，但只是数字恰好一样。

概念上：

$$
\boxed{\text{group size} \neq \text{residual length}}
$$

### 9. KIVI 最终理解

- K：固定一个 K channel，沿 token / sequence 收集，token group 内量化。
- V：固定一个 token，看整个 V vector，per-token 量化。

画成方向就是：

```text
K：

         c3
token0   ↓
token1   ↓
token2   ↓
token3   ↓

沿 token 方向看同一个 channel


V：

token3 → c0 c1 c2 c3 ...

沿 head_dim 看当前 token
```

最终：

> KIVI 是 KV Cache 的低 bit 在线量化方案。K 沿 token 方向分组，并在每个 token group 内做 per-channel quantization，因此不同 group 的相同 K channel 通常拥有不同 qparams；V 主要采用 per-token quantization，同时最新的一部分 KV 作为 residual/recent 保持 FP16。

一句话：

$$
\boxed{\text{KIVI} = K:\ \text{token-group} \times \text{channel};\ \ V:\ \text{per-token} + \text{recent FP16}}
$$

---

## 六、KVQuant

### 1. KVQuant 也是 KV Cache quantization

KVQuant 和 KIVI 解决的都是：KV Cache 太大。

但 KVQuant 比 KIVI 更复杂。它不是只提出 K per-channel、V per-token，而是**组合了多种方法来保证极低 bit KV quantization 的精度**。

### 2. Key 仍然是 per-channel

对于一个 KV head：

$$
K: [T, D]
$$

这里的 channel 仍然是：

$$
D = \text{head\_dim}
$$

例如 `head_dim = 128`：c0, c1, …, c127。

KVQuant 同样认为：Key 使用 per-channel 方向更加合适。

### 3. 和 KIVI 最重要的区别

KIVI 的 K：

```text
当前 token group0 → 根据 group0 自己的数据 → 算 c0 的 qparams
当前 token group1 → 根据 group1 自己的数据 → 重新算 c0 的 qparams
```

所以 group0, c0 → 一套；group1, c0 → 另一套。

KVQuant 则引入 **Calibration dataset**：预先观察 Key 各个 channel 的数值分布，并提前确定量化所需要的统计信息 / range / 参数。

于是 runtime：

```text
token0 的 c0
token1 的 c0
token2 的 c0
...
```

可以直接按照 c0 对应的 **calibration-based statistics** 进行处理。

因此它不依赖 KIVI 那种"必须等当前 sequence 凑够一整个 token group，再根据这一组数据统计 channel qparams"的基本机制。

### 4. "同一列 K 共用 scale"应该怎么准确理解？

我们之前说"所有 K 相同列使用同一套 scale/zero"。这个理解必须加限定。

不是"整个模型的所有 c0 → 一个 scale"，而应该理解到类似：

```text
layer0
  KV head0
    channel c0 → 对应自己的统计

layer0
  KV head1
    channel c0 → 另一套统计

layer1
  KV head0
    channel c0 → 又是一套统计
```

所以理解层级是：

$$
\boxed{\text{layer} \times \text{KV head} \times \text{K channel}}
$$

而不是全模型简单按"列号"共享。

### 5. Value 怎么量化？

Value 仍然更适合 per-token。

因为：

$$
V_{\text{new}} = [v_0, v_1, \dots, v_{127}]
$$

当前 token 产生的时候，一整条 V vector 已经完整，所以不用等待未来 token：

```text
当前 V → 直接根据自己这一条 vector 进行量化
```

### 6. 为什么 KVQuant 还需要其他技巧？

因为目标已经可能来到 KV3 / KV2。2 bit 只有：

$$
2^2 = 4
$$

种 code。如果只是简单 uniform min-max quantization，精度损失很容易过大。

所以 KVQuant 还加入了一系列保护机制。

### 7. 技巧一：Key per-channel

也就是刚才讲的：固定 K channel，沿 sequence 看其统计，利用 Key 的 channel-wise 分布特征。

### 8. 技巧二：RoPE 前量化 Key

正常 Transformer 里 K 会经过 RoPE。KVQuant 发现：**RoPE 会改变 / 混合 Key channel 的数值分布**，使原本适合做 channel-wise quantization 的结构受到影响。

所以它倾向于在 **RoPE 之前的 Key** 上进行量化。

概念流程可以理解成：

```text
K_pre-RoPE
    ↓
quantize / store
    ↓
需要 Attention
    ↓
dequantize
    ↓
RoPE
    ↓
QK^T
```

工程实现需要尽可能把 dequant + RoPE + Attention 融合，避免额外 kernel 开销。

### 9. 技巧三：Non-uniform quantization

普通 uniform INT2：四个量化级别间距基本规则。

但是实际 KV 分布并不一定均匀。如果大量数据集中在 0 附近，却拿四个等距离 code 覆盖整个 range，很多表示能力被浪费。

所以 KVQuant 使用更适合实际数据分布的 **non-uniform quantization**。

核心理解：

> **2 bit 的 code 太少，因此要把有限的 code 用在数据最需要的地方。**

### 10. 技巧四：Dense / Sparse Outlier 分离

例如：

```text
[0.3, 0.5, 0.2, 5.0]
```

如果 5.0 和其他正常值一起做 INT2，range 被 5.0 拉大，导致 0.2 / 0.3 / 0.5 精度非常差。

于是可以：

```text
正常值 → dense low-bit quantization
outlier → sparse 单独处理
```

核心思想：

> **不要让少量 outlier 污染绝大多数正常 KV。**

### 11. 技巧五：Attention Sink

LLM 中有 Attention Sink 现象：某些开头 token（尤其第一个 token）会得到很多后续 token 的 attention。

所以如果第一个 token 的 KV 出现比较大的 quantization error，这个误差可能影响后续大量 attention 计算。

因此 KVQuant 对这种位置进行特殊保护，例如让 sink token 保留较高精度。

核心不是"第一个 token 数值一定最大"，而是 **attention 对它的误差特别敏感**。

### 12. 技巧六：Offline Calibration

对于 K per-channel quantization，如果完全 online：

```text
token0 到来
token1 到来
token2 到来
...
```

每来新的 token，该 channel 的 min/max 都有可能改变，于是 scale 可能一直变化，那已经用旧 scale 量化好的历史 K 就变得麻烦。

所以 KVQuant 使用 calibration data 提前统计 Key channel 的分布。

核心理解：

> **未来真正的 K/V 仍然是在 runtime 产生；所谓 offline 并不是提前知道未来 KV，而是提前确定量化所需要的统计信息和量化策略。**

### 13. KIVI vs KVQuant

最简单可以这样记：

| | KIVI | KVQuant |
|---|---|---|
| K | 当前 sequence 实际数据 → token group → group 内 per-channel | Calibration 提前得到 channel statistics → runtime per-channel |
| V | per-token | per-token |
| 其他 | recent/residual FP16 | pre-RoPE、non-uniform、dense/sparse outlier、Attention Sink… |
| 定位 | 强调 online grouped KV quantization | 强调 calibration + 多种极低 bit 精度保护技巧 |

### 14. KVQuant 最终记忆

> KVQuant 对 K 采用 calibration-based per-channel quantization，这里的 channel 是每个 KV head 的 head_dim feature；V 主要采用 per-token quantization。为了让 KV Cache 能进一步压到极低 bit，它还加入 RoPE 前 Key quantization、non-uniform quantization、dense/sparse outlier 分离、Attention Sink 保护和 offline calibration 等技巧。

一句话：

$$
\boxed{\text{KVQuant} = K\ \text{calibration-based per-channel} + V\ \text{per-token} + \text{多种低 bit 精度保护}}
$$

---

## 七、五个算法统一对比

| 方法 | 量化对象 | 典型精度 | 最核心思想 | "Channel"最重要的含义 |
|---|---|---|---|---|
| GPTQ | Weight | W4A16 | Hessian 指导误差补偿 | Hessian 对应 input-feature / input-channel 关系 |
| AWQ | Weight | W4A16 | Activation 判断并保护重要 Weight | scaling 沿 input channel |
| SmoothQuant | Weight + Activation | W8A8 | 把 Activation 量化难度迁移给 Weight | smoothing 沿 input channel |
| KIVI | KV Cache | KV2 | K per-channel，V per-token | K channel = KV head 的 head_dim |
| KVQuant | KV Cache | KV2 / KV3 等 | calibration + 多种低 bit 保护 | K channel = KV head 的 head_dim |

---

## 八、最后一定要搞清楚的三种 Channel

### 1. Linear output channel

对于 $Y = XW$，就是 $Y$ 的 feature / 输出维度。

Weight per-channel quantization 经常针对：

$$
\boxed{\text{output channel}}
$$

PyTorch layout：**一行 Weight = 一个 output channel。**

### 2. Linear input channel

就是 $X$ 的 hidden / feature 维度，同时也是 Weight 和 X 相乘的维度。

主要出现在：

- GPTQ → Hessian 的 input-feature 维度
- AWQ → activation-aware scaling
- SmoothQuant → smoothing scaling

AWQ / SmoothQuant 的 $X/s$、$W \times s$ 主要就是沿：

$$
\boxed{\text{input channel}}
$$

完成的。

### 3. KV Cache channel

对于：

$$
K, V: [T, D]
$$

这里 $T$ = sequence/token，$D$ = head_dim。

KIVI / KVQuant 说 "Key per-channel"，这里的 channel 指：

$$
\boxed{\text{KV head 的 head\_dim}}
$$

不是 Linear 的 input/output channel。

---

## 九、面试前 30 秒版本

**GPTQ**

> Weight-only PTQ → calibration 得到 Hessian → 量化当前 Weight → 用后续 Weight 补偿误差 → W4A16 / per-group

**AWQ**

> Weight-only PTQ → Activation 告诉我哪些 input channels 重要 → scaling 保护对应 Weight → 再做 W4 per-group

**SmoothQuant**

> Activation outlier 太难 INT8 → 沿 input channel 做 $X/s$、$W \times s$ → 保持 $Y=XW$ 不变 → 把量化难度迁移给 Weight → W8A8

**KIVI**

> KV Cache quantization → K：沿 token 分组，组内 per-channel → V：per-token → 不同 K group 同 channel 的 qparams 不同 → recent/residual FP16

**KVQuant**

> KV Cache quantization → K：calibration-based per-channel → V：per-token → pre-RoPE → non-uniform → dense/sparse outlier → Attention Sink

---

最上层只需要记住：

| 方法 | 一句话记法 |
|---|---|
| GPTQ | 怎么**补偿** Weight quantization error |
| AWQ | 怎么**保护**重要 Weight |
| SmoothQuant | 怎么解决 Activation **难量化** |
| KIVI | 怎么**在线量化** KV Cache |
| KVQuant | 怎么把 KV Cache 压到**极低 bit** 还尽量保持精度 |


# 大模型量化完整学习资料（完整版）

> **定位**：本文是「知乎答主『吃果冻不吐果冻皮』《目前大模型量化方案有很多，有哪些比较SOTA的量化方案？》（2025-01-24，获386赞）主回答 + 学习过程 44 轮深度追问」的完整整理版。
>
> **内容来源**：本文综合整理了 `Quant` 目录下的《大模型量化完整学习资料.md》（精炼版）、`quant.md`（零漏项版）与 `quant_dialogue_raw_v2.md`（历史对话）。保留了**全部量化知识主体 + 每一个方法的完整数值推演 + 学习过程中反复澄清的关键点与易错校正**；仅删除了对话中的非技术杂项（文件导出、IDE 操作、覆盖检查表、整理过程附录等）。
>
> **原文链接**：
> - 博主知乎主页：[吃果冻不吐果冻皮](https://www.zhihu.com/people/liguodong-iot/posts)
> - 主回答/系列概述（知乎）：[大模型量化概述](https://www.zhihu.com/question/627484732/answer/3261671478)
> - 公众号原文：[大模型量化技术原理总结](https://mp.weixin.qq.com/s/E2itzyivEY-dg0O-7sICnw)
> - 更多：博主系列文章索引与本回答涉及的参考文档，见文末[附录：参考文档](#附录参考文档)
>
> **贯穿示例**：全文使用同一组固定的 4×4 数据来推演每个概念与方法，便于对照理解。
>
> ```
> 权重 W(4×4)：
> out0: [0.5, 0.1, 0.3, 6.0]
> out1: [0.4, 0.2, 0.6, 0.8]
> out2: [0.7, 0.3, 0.5, 0.2]
> out3: [0.1, 0.9, 0.4, 0.6]
>
> 激活 X(2×4)：
> token0: [2.0, 0.5, 0.8, 0.3]
> token1: [0.4, 7.0, 0.2, 0.6]
> ```
>
> **标记说明**：
> - `🔍 学习追问`：历史学习中反复确认、容易混淆的知识点及其最终结论；
> - `⚠️ 易错校正`：此前版本中曾混淆、已校正的概念；
> - 正文为博主原文知识，推演与对比为学习过程补充。

---

## 目录

- [第一部分：基本概念](#第一部分基本概念)
  - [1.1 简介](#11-简介)
  - [1.2 模型量化的粒度](#12-模型量化的粒度)
  - [1.3 模型量化对象](#13-模型量化对象)
  - [1.4 静态量化与动态量化](#14-静态量化与动态量化)
  - [1.5 离线量化与在线量化](#15-离线量化与在线量化)
  - [1.6 线性量化与非线性量化](#16-线性量化与非线性量化)
  - [1.7 量化数据类型](#17-量化数据类型)
  - [1.8 量化分类（QAT / QAF / PTQ）](#18-量化分类qat--qaf--ptq)
- [第二部分：模型量化方法精要](#第二部分模型量化方法精要)
  - [2.1 RTN](#21-rtn)
  - [2.2 LLM-QAT](#22-llm-qat)
  - [2.3 QLoRA](#23-qlora)
  - [2.4 LLM.int8()](#24-llmint8)
  - [2.5 GPTQ](#25-gptq)
  - [2.6 SmoothQuant](#26-smoothquant)
  - [2.7 AWQ](#27-awq)
  - [2.8 SpQR](#28-spqr)
  - [2.9 ZeroQuant 系列](#29-zeroquant-系列)
  - [2.10 FP6-LLM](#210-fp6-llm)
  - [2.11 KIVI](#211-kivi)
  - [2.12 IntactKV](#212-intactkv)
  - [2.13 KVQuant](#213-kvquant)
  - [2.14 Atom](#214-atom)
  - [2.15 QuaRot](#215-quarot)
  - [2.16 QoQ](#216-qoq)
  - [2.17 方法总回顾：18 个方法的核心思想一句话总结](#217-方法总回顾18-个方法的核心思想一句话总结)
- [第三部分：量化对大模型的影响](#第三部分量化对大模型的影响)
  - [3.1 量化对模型精度的影响](#31-量化对模型精度的影响)
  - [3.2 量化对不同任务类型能力的影响](#32-量化对不同任务类型能力的影响)
  - [3.3 量化对于模型推理性能的影响](#33-量化对于模型推理性能的影响)
- [第四部分：如何选择正确的量化方法（最佳实践）](#第四部分如何选择正确的量化方法最佳实践)
- [第五部分：其他答主补充](#第五部分其他答主补充)
- [附录：参考文档](#附录参考文档)

---

# 第一部分：基本概念

## 1.1 简介

模型量化是一种用于减少神经网络模型大小和计算量的技术，将模型参数（如：权重）从高精度数据类型（如：float32）转换为低精度数据类型（如：int8 或 fp4）。模型量化通过以更少的位数表示数据，可以减少模型尺寸，进而减少在推理时的内存消耗，并且在一些低精度运算较快的处理器上可以增加推理速度，同时仍然可以保持模型的性能。

近年来，随着 Transformer、[MOE架构](https://zhida.zhihu.com/search?content_id=710333991&content_type=Answer&match_order=1&q=MOE%E6%9E%B6%E6%9E%84&zhida_source=entity) 的提出，深度学习模型轻松突破上万亿规模参数，模型压缩变得至关重要。模型压缩主要分为：

- [模型剪枝](https://zhida.zhihu.com/search?content_id=710333991&content_type=Answer&match_order=1&q=%E6%A8%A1%E5%9E%8B%E5%89%AA%E6%9E%9D&zhida_source=entity)（Pruning）
- [知识蒸馏](https://zhida.zhihu.com/search?content_id=710333991&content_type=Answer&match_order=1&q=%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F&zhida_source=entity)（Knowledge Distillation）
- **模型量化**

> 🔍 **全文逻辑主线**：量化的核心矛盾是「离群值（outlier）拉大量化步长 → 中间值精度被牺牲」。后面所有方法本质上都是在回答同一个问题：**如何先处理 outlier / sensitivity，再让简单的量化也能工作？**
> - RTN：什么都不处理，直接 round → 精度最差
> - SmoothQuant/AWQ：把激活的离群值"压"下去（数学等价变换）→ 再量化
> - GPTQ：量化后调整其他权重来补偿误差（优化视角）
> - LLM.int8()：把离群值单独拎出来用 FP16 算
> - QuaRot：用旋转矩阵把离群值"稀释"掉

---

**博主原文配图**（源自知乎原文）：

![原文图 1](https://picx.zhimg.com/v2-c4f12ac6e06a00e5bd24d73e2807a2bd_r.jpg?source=2c26e567)


![原文图 2](https://picx.zhimg.com/v2-d493f0bec79b92fe813de0a730e67a66_r.jpg?source=2c26e567)


---

### 📌 原始 Q&A：学习设定与固定例子（历史第 2-6 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 2 轮）


---
阅读完全了吗？对于这个完整的？

##### A（历史第 3 轮，正文原样保留；仅将内部标题降级以保持目录层级）

是的，我已经读取完了该 HTML 文件的全部文本内容。不过需要注意的是，这个页面一共列出了 **18 个回答**，但目前提取到的完整回答只有 **3 个**：

1. **吃果冻不吐果冻皮** — 主答，完整的大模型量化技术综述（最长）
2. **Ironboy** — 补充了2025年3月的SOTA更新
3. **不归牛顿管的熊猫** — about any4 (Meta FAIR ICML 2025)

其余15个回答可能在HTML中因懒加载未能完全获取（知乎页面通常会折叠非高赞回答）。

另外我看到你打开了 [baidu_question.md](Infra/baidu_question.md)，需要我阅读那个文件吗？还是你想基于这篇知乎文章的内容做进一步的事情（比如，提炼笔记、对比方案、翻译要点等）？

##### Q（历史第 4 轮）

我们主要来讨论关于 吃果冻不吐果冻皮 的回答

##### A（历史第 5 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好的，那我们就聚焦 **吃果冻不吐果冻皮** 的这篇回答。我读取到的内容已经包含了该回答的完整章节，从"基本概念"到"结语"。

以下是这篇回答的完整结构回顾：

---

###### 📋 文章全貌

| 章节 | 内容概要 |
|------|----------|
| **基本概念** | 量化粒度、量化对象（Weight/Activation/KV Cache/Gradient）、静态vs动态量化、离线vs在线量化、线性vs非线性量化、量化数据类型 |
| **RTN** | 四舍五入，最简单但受离群值影响大 |
| **LLM-QAT** | QAT + 无数据知识蒸馏，对称MinMax |
| **QLoRA** | NF4 + 双量化 + 分页优化器 |
| **LLM.int8()** | 混合精度分解，W8A8 |
| **GPTQ** | 优化视角，Hessian矩阵，W4A16 |
| **SmoothQuant** | 平滑因子转移量化难度，W8A8 |
| **AWQ** | 保护1%显著权重，激活感知，W4A16 |
| **SpQR** | 隔离异常值 + 双层量化元数据 |
| **ZeroQuant系列** | 逐Token动态量化 + LoRC补偿 |
| **FP6-LLM** | TC-FPx kernel，W6A16 |
| **KIVI** | KV缓存2bit量化，key逐通道/value逐Token |
| **KVQuant** | 离线校准 + RoPE前量化 + Attention Sink |
| **Atom** | W4A4KV4，混合精度+通道重排 |
| **QuaRot** | Hadamard旋转消除异常值，W4A4KV4 |
| **QoQ** | 渐进式分组量化 + SmoothAttention，W4A8KV4 |
| **量化影响分析** | 精度、不同任务类型（NLP/涌现能力/对话/长文本）、推理性能 |
| **最佳实践** | TensorRT-LLM选型指南 |

---

你对这篇回答的哪个部分感兴趣？比如：

1. **某个具体的量化方法** — 想深入了解 GPTQ / AWQ / SmoothQuant 等的原理？
2. **量化选型建议** — 结合你自己的部署场景（什么模型、什么硬件、什么batch size）帮你选？
3. **量化对模型能力的影响** — 精度损失、涌现能力丧失等？
4. **笔记/提炼** — 帮你把这篇整理成更精炼的笔记或思维导图？

或者你打开的 [baidu_question.md](Infra/baidu_question.md) 和这个话题有关？

##### Q（历史第 6 轮）

我们尽量以一个具体的transformer层的例子数据来进行讲解你觉得呢？

##### A（历史第 7 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好主意！用一个具体的小例子来演示，比纯理论直观得多。我先提出一个设计方案，你看看合不合适。

---

###### 建议方案

设计一个 **微型Transformer层**（极度简化版），比如：

###### 示例数据

```
隐藏层维度: d = 8
序列长度:    T = 4
```

- **权重矩阵 W** (8×8)：故意加入几个离群值（比如在第3、第7列设置大的数值）
- **激活输入 X** (4×8)：同样引入一些异常值分布
- **KV Cache**: 简单模拟 K 和 V 矩阵

###### 演示的量化方法（按递进难度）

| 方法 | 演示要点 |
|------|----------|
| **① RTN (Round to Nearest)** | 四舍五入量化 W4A16，看 baseline 误差 |
| **② SmoothQuant** | 计算平滑因子 s，把激活的量化难度转移到权重 |
| **③ AWQ** | 根据激活分布找显著通道，grid search 最佳 α |
| **④ GPTQ** | 单层优化视角 + Hessian 矩阵 + 逐组量化 |
| **⑤ LLM.int8()** | 混合精度分解，离群值分离 |

###### 输出对比

逐一计算量化后的 **输出误差**（MSE / 余弦相似度），在同一个表格里对比。

---

你觉得这个规模合适吗？或者你希望：

- **更小**（比如 d=4, T=2）方便手算？
- **再加一个方法**（比如 KIVI 的 KV 缓存量化）？
- **直接用 Python 跑**，看实际的数值变化和误差对比？

另外，你打开的 [baidu_question.md](Infra/baidu_question.md) 里是不是有相关的背景信息？需要我一起读吗？

## 1.2 模型量化的粒度

量化粒度决定了**一组参数共享同一个缩放因子（scale）和零点（zero point）的范围大小**。粒度越细，每组内数值分布越集中，量化精度越高，但存储的量化参数（s, z）越多。

四种粒度：

- **per-tensor（又名 per-layer）量化**：每层或每个张量只有一个缩放因子，张量内的所有值都被这个缩放因子量化。
- **per-channel 量化**：卷积核的每个通道都有不同的缩放因子。
- **per-token 量化**：针对激活而言，针对每一行进行量化。在 LLM 中，通常与 per-channel 量化搭配使用，如：逐 Token 量化激活，逐通道量化权重。
- **per-group / group-wise**：以组为单位。将每个密集矩阵按输出神经元进行分割，每个连续的 N 输出神经元作为一个组。比如：GPTQ、AWQ 中使用 128 个元素为一组进行量化。有些地方也称为子通道分组（Sub-channel-wise）量化。

> 🔍 **博主原文补充**：如 **Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT** 中所说，分组量化的一种特殊情况是**将每个密集矩阵视为一组**，每个矩阵都可以有自己的量化范围；而更普遍的情况是**将每个密集矩阵按输出神经元进行分割，每个连续的 N 输出神经元作为一个组**。

> ⚠️ **group-wise 的权衡**：分组越细，scale 越精准，但存储 scale 的元数据开销越大，而且可能出现数值溢出 INT8 范围需要 clip 的情况。下面是 8×8 矩阵的分组权衡示例：
>
> | Group size | Scale 数量 | 元数据开销 | 量化精度 |
> |:----------:|:----------:|:----------:|:--------:|
> | per-tensor（1 组） | 1 | 最低 | 最差 |
> | per-channel（8 组） | 8 | 低 | 中等 |
> | group=4（16 组） | 16 | 中 | 高 |
> | group=2（32 组） | 32 | 高 | 更高 |
> | per-element（64 组） | 64 | 最高 | 最好 |

> 🔍 **学习追问：量化粒度在逻辑上到底在问什么？**
>
> 量化粒度的本质 = **共享 scale/z 的范围大小**。
>
> - per-tensor：整个矩阵共享一组 (s, z) → 范围最大，精度最低，参数最少
> - per-channel：每一列（输出通道）共享一组 (s, z)
> - per-group：每 N 个连续元素共享一组 (s, z)（如 group=128）
> - per-token：每一行（每个 token 的激活）共享一组 (s, z)
> - per-element：每个元素单独一套 (s, z)——逻辑上无意义（量化自己再反量化自己，无误差也无压缩）

> 🔍 **粒度递进的直观类比**
>
> - **per-tensor** = 全班同学穿同一码校服（高个子拉高了平均，矮子穿得晃荡）
> - **per-channel** = 按身高分男女各穿各码（更合理）
> - **per-group** = 按身高每 5cm 一档定制（更贴身）
> - **per-element** = 每个人单独量体定制（最合身，但没必要）

### 用数据推演四种粒度

使用贯穿示例：权重矩阵 4×4，激活矩阵 2×4。

**① Per-tensor：整个矩阵 16 个元素共用 1 个 (s,z)**

```
W = [0.5, 0.1, 0.3, 6.0, 0.4, 0.2, 0.6, 0.8, 0.7, 0.3, 0.5, 0.2, 0.1, 0.9, 0.4, 0.6]
    min=0.1                                   max=6.0

全局 r_max = 6.0, r_min = 0.1
步长 s = (6.0 - 0.1) / 255 ≈ 0.0231

out1 行本来的值 [0.4, 0.2, 0.6, 0.8] 都在 0.1~0.8 范围，却被迫用一个 0.0231 的粗步长。
```

**② Per-channel（按行/输出通道）：每行 4 个元素共用 1 个 (s,z)**

```
out0: [0.5, 0.1, 0.3, 6.0]  → s₀ = 5.9/255 = 0.0231  ← 被自己的 6.0 拖累
out1: [0.4, 0.2, 0.6, 0.8]  → s₁ = 0.6/255 = 0.0024  ← 精细！
out2: [0.7, 0.3, 0.5, 0.2]  → s₂ = 0.5/255 = 0.0020  ← 精细！
out3: [0.1, 0.9, 0.4, 0.6]  → s₃ = 0.8/255 = 0.0031  ← 精细！
```

行1~行3 摆脱了行0那个 6.0 的污染，步长变精细；但行0自己仍被 6.0 拖累。

**③ Per-group（行内再分割）：每行分成 2 组，每组 2 个元素共用 1 个 (s,z)**

```
out0: [0.5, 0.1 | 0.3, 6.0]     ← 组A₀: s=0.4/255=0.0016 精细！
                                     组B₀: s=5.7/255=0.0224 粗
```

行0的 6.0 被隔离在组B₀，不再污染组A₀；[0.5, 0.1] 几乎无损量化。**离群值只影响它所在的局部小组，其他小组不受牵连。**

**④ Per-token（激活专用）：每行（=每个 token）4 个元素共用 1 个 (s,z)**

```
X 激活：
token0: [2.0, 0.5, 0.8, 0.3]  → s₀ = 1.7/255 = 0.0067 ← 精度提升！
token1: [0.4, 7.0, 0.2, 0.6]  → s₁ = 6.8/255 = 0.0267 ← 被自己的 7.0 拖累

如果 per-tensor：整个 X 用 s = (7.0-0.2)/255 = 0.0267，token0 全用粗步长。
```

**完整对比（用我们的 4×4 数据）**

| 粒度 | 组数 | 组大小 | 受影响最严重的组 | 最精细的组 | 核心逻辑 |
|:----:|:----:|:------:|:----------------:|:----------:|:--------|
| **Per-tensor** | 1 | 16个元素 | 唯一一组，小值被 6.0 拖累 | — | 全校服一个码 |
| **Per-channel** | 4 | 每行4个 | 行0: s=0.0231 | 行2: s=0.0020 | 按身高分男女码 |
| **Per-group** | 8 | 每组2个 | 行1组B: s=0.0224 | 行0组A: s=0.0016 | 按身高每5cm一档 |
| **Per-token(激活)** | 2 | 每行4个 | token1: s=0.0267 | token0: s=0.0067 | token不被其他token污染 |

**核心一句话**：粒度越细，离群值的"污染半径"越小。

> ⚠️ **易错校正 1：不是所有"per-channel"都指同一个方向**
>
> - 对**权重**：经常指 **per-output-channel**（每一行）；
> - 对**激活**：可能指 hidden/input channel，即 `X[:,j]`；
> - 对 **K Cache**：KIVI 里的 per-channel 是跨 token、沿同一个 hidden channel 量化。
>
> 听到"per-channel"时第一反应不能只是"按列"，而应问：**这个 tensor 的 channel 维到底是哪一维？qparams 是沿哪一维共享的？**

> ⚠️ **易错校正 2："per-channel 与 INT8 GEMM 不兼容"过于宽泛**
>
> 更准确地说：SmoothQuant 论文讨论的是 **activation 在 GEMM 内积维度上的 per-channel scaling** 不容易直接映射到高吞吐 INT8 GEMM；而 **weight 的 per-output-channel scale 位于 GEMM 外维度**，通常可以高效处理。

> ⚠️ **易错校正 3：GPTQ 中非常容易混淆的两个"128"**
>
> 1. **quantization group_size=128**：决定哪些权重共享 scale/zero；
> 2. **GPTQ algorithm block_size B=128**：Lazy Batch Update 一次处理多少列。
>
> 它们可能都等于 128，但概念完全不同。

**各主流方法采用的量化粒度**

| 量化方法 | 权重量化粒度 | 激活量化粒度 | KV缓存量化粒度 |
|:--------:|:-----------:|:-----------:|:-------------:|
| **GPTQ** | per-group (g=128) | — (FP16) | — |
| **AWQ** | per-group (g=128) | — (FP16) | — |
| **SmoothQuant** | per-channel / per-tensor | per-token / per-tensor | — |
| **LLM.int8()** | per-channel | per-token | — |
| **ZeroQuant** | per-group (g=64/128) | per-token (动态) | — |
| **KIVI** | — | — | per-channel(K) + per-token(V) |
| **QuaRot** | per-group | per-token | per-group |
| **QoQ** | per-channel + per-group | per-token | per-group |

---

### 📌 原始 Q&A：模型量化的粒度（历史第 8-12 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 8 轮）

现在我们学 该答主讲的。模型量化的粒度

##### A（历史第 9 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好的！我们从最基础的开始。先设定一个具体的 **Transformer 层示例数据**，然后逐个粒度讲解。

---

###### 🧱 设定示例

假设一个简化 Transformer 层的 **权重矩阵** 和 **激活矩阵**：

```
隐藏层维度 d = 8
输入 Token 数 T = 4
```

**权重矩阵 W**（形状 8×8，输出维度×输入维度）：

```
W[8×8] = 
行/输出通道 →   列/输入通道
              c0   c1   c2   c3   c4   c5   c6   c7
   out0:     [0.5, 0.8, 0.1, 0.3, 0.4, 0.2, 0.6, 0.9]
   out1:     [0.7, 0.2, 0.5, 0.4, 0.3, 0.1, 0.8, 0.6]
   out2:     [0.3, 0.6, 0.9, 0.2, 0.1, 0.7, 0.4, 0.5]
   out3:     [0.8, 0.1, 0.4, 0.7, 0.6, 0.3, 0.2, 0.5]
   out4:     [0.2, 0.5, 0.8, 0.6, 0.9, 0.4, 0.1, 0.3]
   out5:     [0.6, 0.3, 0.2, 0.1, 0.7, 0.8, 0.5, 0.4]
   out6:     [0.4, 0.7, 0.6, 0.9, 0.2, 0.5, 0.3, 0.8]
   out7:     [0.1, 0.4, 0.3, 0.5, 0.8, 0.9, 0.7, 0.2]
```

**权重列 c3 的离群值**（红色）：c3列整体偏大，最大值 0.9

---

###### 1️⃣ Per-tensor 量化（又名 Per-layer）

**定义**：整个张量（整个权重矩阵）**只用一个缩放因子 s**

```
计算：
  r_max = W 中的最大值 = 0.9
  r_min = W 中的最小值 = 0.1
  s = (0.9 - 0.1) / (127 - (-128))   ← 若用 INT8 对称量化
    = 0.8 / 255 ≈ 0.00314
```

**问题**：如果 W 大部分值集中在 [0.1, 0.5]，但因为 c3 列有 0.9 的离群值，步长 s 被"拉大"了。所有值都被"压缩"到更粗的格子里，非离群值的量化误差变大。

```
量化后的误差（示意）：
  0.2 → round(0.2/0.00314) = round(63.7) = 64 → 反量化 64×0.00314 = 0.201 ✅ 误差小
  0.1 → round(0.1/0.00314) = round(31.8) = 32 → 反量化 32×0.00314 = 0.100 ❌ 几乎不变
  但若大多数值在 0.1-0.5 范围，0.9 的存在让整个分辨率变差
```

**优点**：实现最简单，硬件效率最高（一个scale值）
**缺点**：受离群值影响大，精度损失显著

---

###### 2️⃣ Per-channel 量化

**定义**：对权重矩阵的 **每个输出通道（每一行）** 各自算一个缩放因子 s

```
对 out0 行: r_max=0.9, r_min=0.1, s₀=(0.9-0.1)/255=0.00314
对 out1 行: r_max=0.8, r_min=0.1, s₁=(0.8-0.1)/255=0.00275
对 out2 行: r_max=0.9, r_min=0.1, s₂=(0.9-0.1)/255=0.00314
对 out3 行: r_max=0.8, r_min=0.1, s₃=(0.8-0.1)/255=0.00275
...

实际计算以 out1 行为例：
  0.7 → round(0.7/0.00275) = round(254.5) = 255 → 反量化 0.701 ✅ 误差较小
  0.2 → round(0.2/0.00275) = round(72.7) = 73  → 反量化 0.201 ✅ 
```

**优点**：每行有独立的 scale，避免"一行的大值污染其他行"  
**缺点**：无法直接映射到 INT8 GEMM Kernel（硬件不高效）

> 答主原文提到："per-channel 保留了精度，但它与 INT8 GEMM Kernel 不兼容"

---

###### 3️⃣ Per-token 量化（主要针对激活）

**定义**：对激活矩阵，**每个 token（每一行）** 各自量化

假设激活输入 X（4个token，每个token 8维）：

```
X[4×8] = 
         c0   c1   c2   c3   c4   c5   c6   c7
token0: [1.2, 0.5, 0.8, 5.3, 0.4, 0.2, 0.6, 0.9]  ← c3有异常值！
token1: [0.3, 0.7, 0.2, 0.4, 0.6, 0.1, 0.5, 0.8]
token2: [4.8, 0.3, 0.6, 0.9, 0.2, 0.7, 0.1, 0.5]  ← c0有异常值！
token3: [0.5, 0.4, 0.3, 0.2, 0.7, 0.6, 0.8, 0.1]
```

```
Per-tensor 量化激活（统一一个scale）：
  r_max = 5.3, r_min = 0.1
  s_all = (5.3-0.1)/255 = 0.0204
  token1所有值都在0.1-0.8之间，但用一个0.0204的步长 → 精度受损严重！

Per-token 量化激活（每个token一个scale）：
  s_token0 = (5.3-0.1)/255 = 0.0204  ← token0有离群值5.3，没办法
  s_token1 = (0.8-0.1)/255 = 0.00275 ← token1正常范围，步长小！精度高！
  s_token2 = (4.8-0.1)/255 = 0.0184  ← token2也有离群值
  s_token3 = (0.8-0.1)/255 = 0.00275 ← token3正常，精度高
```

**关键 insight**：Per-token 量化保护了那些"没有离群值的 token"不被污染

---

###### 4️⃣ Per-group 量化（Group-wise）

**定义**：在每行内部 **再分小组**，每组一个 scale。答主原文说"GPTQ、AWQ中使用128个元素为一组"

用我们 8×8 的矩阵，假设 group size = **4 列一组**：

```
out0行 [0.5, 0.8, 0.1, 0.3 | 0.4, 0.2, 0.6, 0.9]
         ←group0, s₀→      ←group1, s₁→

group0: r_max=0.8, r_min=0.1, s₀=(0.8-0.1)/255=0.00275
  0.5 → round(0.5/0.00275)=round(181.8)=182 → 反量化 0.501 ✅
  0.1 → round(0.1/0.00275)=round(36.4)=36   → 反量化 0.099  ✅

group1: r_max=0.9, r_min=0.2, s₁=(0.9-0.2)/255=0.00275
  0.9 → round(0.9/0.00275)=round(327.3) → INT8最大127 → clip到127 → 反量化 0.349 ❌
  !! 0.9超过INT8范围，被截断了
  0.4 → round(0.4/0.00275)=round(145.5) → clip到127 → 反量化 0.349 ❌
```

这里暴露了 **group-wise 的核心问题**：分组越细，scale 越精准，但存储 scale 的元数据开销越大，而且可能出现数值溢出 INT8 范围需要 clip。

**分组大小对精度和开销的权衡**：

| Group size | Scale 数量（8×8矩阵中） | 元数据开销 | 量化精度 |
|:----------:|:------------------------:|:----------:|:--------:|
| per-tensor (1组) | 1 | 最低 | 最差 |
| per-channel (8组) | 8 | 低 | 中等 |
| **group=4 (16组)** | **16** | **中** | **高** |
| group=2 (32组) | 32 | 高 | 更高 |
| per-element (64组) | 64 | 最高 | 最好 |

---

###### 📊 四种粒度对比总结（用我们的数据模拟）

| 粒度 | 本例中的 scale 数量 | 优点 | 缺点 | 典型应用 |
|:----:|:-------------------:|:----|:----|:--------|
| **Per-tensor** | 1 | 硬件效率最高 | 离群值拉大步长，精度最差 | 简单场景 |
| **Per-channel** | 8 | 精度较好，每行独⽴ | 不与INT8 GEMM兼容 | 权重量化 |
| **Per-token** | 4（按激活行） | 保护无离群值token | 运行时需动态计算，有开销 | 激活量化（ZeroQuant） |
| **Per-group** | 16（group=4） | 精度最高，灵活 | scale 元数据多，可能有clip | GPTQ、AWQ |

---

###### 🔑 关键结论

> **答主原文配图**：不同量化方法中量化对象的粒度选择

| 量化方法 | 权重量化粒度 | 激活量化粒度 | KV缓存量化粒度 |
|:--------:|:-----------:|:-----------:|:-------------:|
| **GPTQ** | per-group (g=128) | — (FP16) | — |
| **AWQ** | per-group (g=128) | — (FP16) | — |
| **SmoothQuant** | per-channel / per-tensor | per-token / per-tensor | — |
| **LLM.int8()** | per-channel | per-token | — |
| **ZeroQuant** | per-group (g=64/128) | per-token (动态) | — |
| **KIVI** | — | — | per-channel(K) + per-token(V) |
| **QuaRot** | per-group | per-token | per-group |
| **QoQ** | per-channel + per-group | per-token | per-group |

---

你觉得这个讲解方式和示例数据的规模合适吗？如果 OK，我们就继续往下走 **"模型量化对象"**——用同样的 W 和 X 数据来演示 Weight / Activation / KV Cache 各自量化的差异和挑战？

##### Q（历史第 10 轮）

可不可以先不按照存储的来看，仅仅是按照逻辑上的？

##### A（历史第 11 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好问题！我们完全抛开存储、硬件、GEMM kernel 这些概念，只从 **逻辑上**看"哪些值共用同一个量化范围"。

---

###### 逻辑本质：量化粒度 = 共享范围的范围大小

量化一个矩阵，本质上只有一个问题：

> **哪些元素共用一个 (s, z) 量化参数？**

---

###### 假设我们有一个矩阵

```
W = [W₀₀, W₀₁, W₀₂, W₀₃]
    [W₁₀, W₁₁, W₁₂, W₁₃]
    [W₂₀, W₂₁, W₂₂, W₂₃]
    [W₃₀, W₃₁, W₃₂, W₃₃]
```

4×4 矩阵，16 个元素。我们从粗到细看：

---

###### ① Per-tensor（最粗）

**整个矩阵 16 个元素共用 1 个 (s, z)**

```
            ┌──────────────────────────────┐
            │  W₀₀ W₀₁ W₀₂ W₀₃           │
            │  W₁₀ W₁₁ W₁₂ W₁₃  ← 共用   │
            │  W₂₀ W₂₁ W₂₂ W₂₃  同一套   │
            │  W₃₀ W₃₁ W₃₂ W₃₃  (s,z)   │
            └──────────────────────────────┘
```

**量化时**：扫一遍所有 16 个数，找到全局 min 和 max，算出 1 个 s 和 1 个 z，然后所有 16 个数都按这个 (s,z) 来映射。

**逻辑问题**：如果矩阵中 15 个数在 [0, 1] 之间，只有 1 个数是 100，那么步长 s 被拉得很大，前 15 个数的精度全部牺牲。

---

###### ② Per-channel（按行/按列分组）

**每一行（或每一列）共用 1 个 (s, z)**

```
按行分（per-output-channel）：
  ┌──────────────────────────────┐
  │  W₀₀ W₀₁ W₀₂ W₀₃  ← 行0共用 (s₀,z₀) │
  │  W₁₀ W₁₁ W₁₂ W₁₃  ← 行1共用 (s₁,z₁) │
  │  W₂₀ W₂₁ W₂₂ W₂₃  ← 行2共用 (s₂,z₂) │
  │  W₃₀ W₃₁ W₃₂ W₃₃  ← 行3共用 (s₃,z₃) │
  └──────────────────────────────┘
```

**量化时**：逐行扫一遍，每行独立算 min/max，得到 4 个 (s, z)。行0中的最大值不会影响行1的步长。

**逻辑意义**：在 Transformer 中，权重矩阵的"行"对应输出神经元。每行覆盖的是一组不同的输出特征，这些特征可能数值范围差异很大。per-channel 让每行"自扫门前雪"。

**按列分（per-input-channel）** 同理，但逻辑意义不同。

---

###### ③ Per-token（激活专用）

**每一行共用 1 个 (s, z)，但"行"被视为一个 token**

假设激活矩阵 X（4 个 token，每个 token 4 维）：

```
X = [x₀₀, x₀₁, x₀₂, x₀₃]  ← token0共用 (s₀,z₀)
    [x₁₀, x₁₁, x₁₂, x₁₃]  ← token1共用 (s₁,z₁)
    [x₂₀, x₂₁, x₂₂, x₂₃]  ← token2共用 (s₂,z₂)
    [x₃₀, x₃₁, x₃₂, x₃₃]  ← token3共用 (s₃,z₃)
```

**逻辑上 per-token 和 per-channel 按行分一模一样**，只是"行"叫法不同。但 per-token 的"行"是 token 维度，per-channel 的"行"是输出通道维度。

**为什么需要单独叫 per-token？** 因为激活矩阵的"行"代表的是序列中的不同 token。不同 token 的数值范围可能差异很大（比如某个 token 刚好是"注意力汇聚"的峰值），per-token 让每个 token 不被其他 token 的离群值影响。

---

###### ④ Per-group（组内更细）

**在行内部再切分，每连续 k 个元素共用 1 个 (s, z)**

假设 group size = 2：

```
  ┌──────────────────────────────┐
  │ [W₀₀,W₀₁]  [W₀₂,W₀₃]  ← 行0分成2组，各用 (s₀₀,z₀₀) (s₀₁,z₀₁) │
  │ [W₁₀,W₁₁]  [W₁₂,W₁₃]  ← 行1分成2组                            │
  │ [W₂₀,W₂₁]  [W₂₂,W₂₃]                                          │
  │ [W₃₀,W₃₁]  [W₃₂,W₃₃]                                          │
  └──────────────────────────────┘
```

**逻辑上**：整个矩阵被分成 4 行 × 2 组 = 8 个独立小组，每个组 2 个元素共用 (s, z)。

**量化时**：每个小组只在这 2 个元素内部找 min/max，算出自己的 (s, z)。相邻的 2 个元素就算数值范围差异很大，也不影响其他组。

**为什么是"连续"的？** 因为逻辑上我们认为相邻的权重元素倾向于有相似的数值分布。如果随机打乱分组，就失去了意义。

---

###### ⑤ 最细粒度：Per-element（每个元素独立）

**每个元素拥有自己的 (s, z)**

```
  ┌──────────────────────────────┐
  │ W₀₀(s₀₀) W₀₁(s₀₁) W₀₂(s₀₂) W₀₃(s₀₃) │
  │ W₁₀(s₁₀) W₁₁(s₁₁) W₁₂(s₁₂) W₁₃(s₁₃) │
  │ ...                                   │
  └──────────────────────────────┘
```

**但逻辑上这是没有意义的！** 因为每个元素只存自己一个值，用自己算出的 (s,z) 再量化自己，等于没有量化——我量化我自己，再反量化，我永远是我自己，没有任何误差，但也没有压缩。

---

###### 一张图看清逻辑层级

```
最粗 ────────────────────────────────────────────────────→ 最细
per-tensor → per-channel/per-token → per-group → per-element
                                                         
共用 (s,z) 越少          ──→          共用 (s,z) 越多
量化步长越"照顾局部"     ──→          量化步长越"精准"
量化误差越小             ──→          量化误差趋近于0
```

---

###### 用我们的 8×8 矩阵看逻辑差异

```
W 中 c3 列整体偏大（0.9 那行附近），其他列偏小（0.1~0.5）

per-tensor 的逻辑：
  "我不管哪行哪列，所有16个数一起看，全局 min=0.1, max=0.9"
  → 步长被拉大，小数值精度受损

per-channel（按行）的逻辑：
  "每行自治，行0的 0.9 不影响行1的 0.1"
  → 行0的步长被行内的0.9拉大，但行1的步长可以很小

per-group（按列分组）的逻辑：
  "行0内部，c0-c3 一组，c4-c7 一组"
  → 0.9 在 c6列，不在 c0-c3组，所以 c0-c3 组可以保持精细步长
```

---

###### 一个类比

- **per-tensor** = 全班同学穿同一码校服（高个子拉高了平均，矮子穿得晃荡）
- **per-channel** = 按身高分男女各穿各码（更合理）
- **per-group** = 按身高每 5cm 一档定制（更贴身）
- **per-element** = 每个人单独量体定制（最合身，但没必要）

---

这样讲，抛开存储和硬件，纯逻辑上你能感受到这种"粒度递进"的脉络吗？如果清楚了，我们继续往下走 **"量化对象"**（Weight vs Activation vs KV Cache 在逻辑上的本质区别）？

##### Q（历史第 12 轮）

我们这样来看，假设token2个，然后hidden——dim=4；这样权重W的都是4*4的；以这个例子给我讲述吧

##### A（历史第 13 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好的！就用这个更精简的设置，数据更清晰。

###### 设定

```
序列长度 T = 2 (token0, token1)
隐藏维度 d = 4
权重矩阵 W: 4×4
激活矩阵 X: 2×4
```

###### 数据

**权重矩阵 W（4×4），每一行是一个输出通道：**

```
    输入通道 →   c0   c1   c2   c3
输出通道 out0:  [0.5, 0.1, 0.3, 6.0]   ← c3 有一个明显离群值 6.0
输出通道 out1:  [0.4, 0.2, 0.6, 0.8]
输出通道 out2:  [0.7, 0.3, 0.5, 0.2]
输出通道 out3:  [0.1, 0.9, 0.4, 0.6]
```

**激活矩阵 X（2×4），每一行是一个 token：**

```
      c0   c1   c2   c3
token0: [2.0, 0.5, 0.8, 0.3]   ← 正常范围
token1: [0.4, 7.0, 0.2, 0.6]   ← c1 有一个离群值 7.0
```

---

###### 逻辑上：量化粒度在问什么？

> **"哪些元素共享同一个量化范围 (s, z)？"**

我们穷尽所有粒度，看同一份数据被怎么分组。

---

###### ① Per-tensor：整个矩阵 16 个元素共用 1 个 (s,z)

```
W = [0.5, 0.1, 0.3, 6.0, 0.4, 0.2, 0.6, 0.8, 0.7, 0.3, 0.5, 0.2, 0.1, 0.9, 0.4, 0.6]
    ↑                                   ↑
    min=0.1                            max=6.0
                             全部共用 (s,z)
```

**发生了什么？**
```
全局 r_max = 6.0   全局 r_min = 0.1
步长 s = (6.0 - 0.1) / (INT8范围) = 5.9 / 255 ≈ 0.0231

out1 行本来的值 [0.4, 0.2, 0.6, 0.8] 都在 0.1~0.8 范围
却用一个 0.0231 的步长去量化
```

这就像：班上身高最矮 150cm，最高 200cm——校服做 5cm 一档。结果 150~160 的同学穿得松松垮垮。**就因为有一个人 200cm，所有人都跟着用粗档位。**

---

###### ② Per-channel（按行 / 按输出通道）：每行 4 个元素共用 1 个 (s,z)

```
out0: [0.5, 0.1, 0.3, 6.0]  ← 共用 (s₀,z₀)
out1: [0.4, 0.2, 0.6, 0.8]  ← 共用 (s₁,z₁)
out2: [0.7, 0.3, 0.5, 0.2]  ← 共用 (s₂,z₂)
out3: [0.1, 0.9, 0.4, 0.6]  ← 共用 (s₃,z₃)
```

**各自算范围：**
```
行0: r_max=6.0, r_min=0.1 → s₀=5.9/255=0.0231
行1: r_max=0.8, r_min=0.2 → s₁=0.6/255=0.0024 ← 精细！
行2: r_max=0.7, r_min=0.2 → s₂=0.5/255=0.0020 ← 精细！
行3: r_max=0.9, r_min=0.1 → s₃=0.8/255=0.0031 ← 精细！
```

**逻辑效果：**
- 行1~行3 摆脱了行0那个 6.0 的污染，它们的步长变得很精细（0.0020~0.0031）
- 但行0自己还是被自己的 6.0 拖累了，步长仍然粗（0.0231）

**问题：** 行0的"好孩子" 0.1, 0.3, 0.5 是和 6.0 这个"坏孩子"分在同一组的，它们也被迫用粗步长。

---

###### ③ Per-group（在行内再分割）：每行分成 2 组，每组 2 个元素共用 1 个 (s,z)

以 group size = 2 为例：

```
out0: [0.5, 0.1 | 0.3, 6.0]     ← 组A₀(s₀₀)    组B₀(s₀₁)
out1: [0.4, 0.2 | 0.6, 0.8]     ← 组A₁(s₁₀)    组B₁(s₁₁)
out2: [0.7, 0.3 | 0.5, 0.2]     ← 组A₂(s₂₀)    组B₂(s₂₁)
out3: [0.1, 0.9 | 0.4, 0.6]     ← 组A₃(s₃₀)    组B₃(s₃₁)
```

**重点看出行0：**
```
组A₀: [0.5, 0.1] → r_max=0.5, r_min=0.1 → s₀₀=0.4/255=0.00157 ← 精细！
组B₀: [0.3, 6.0] → r_max=6.0, r_min=0.3 → s₀₁=5.7/255=0.0224  ← 粗
```

**逻辑效果：**
- 行0的 6.0 被"隔离"在组B₀，不再污染组A₀！
- 组A₀中 [0.5, 0.1] 使用了精细的步长 0.00157，几乎无损量化！

这比 per-channel 又进了一步——**离群值只影响它所在的局部小组，其他小组不受牵连。**

---

###### ④ Per-token（激活专用）：每行（=每个token）4个元素共用 1 个 (s,z)

对激活矩阵 X（2×4）：

```
token0: [2.0, 0.5, 0.8, 0.3]  ← 共用 (s₀,z₀)
token1: [0.4, 7.0, 0.2, 0.6]  ← 共用 (s₁,z₁)
```

**如果改成 per-tensor（整个激活矩阵统一）：**
```
X = [2.0, 0.5, 0.8, 0.3, 0.4, 7.0, 0.2, 0.6]
    min=0.2                     max=7.0
 s = (7.0-0.2)/255 = 0.0267 ← token0 的值 0.3~2.0 全用粗步长
```

**per-token 后：**
```
token0: r_max=2.0, r_min=0.3 → s₀=1.7/255=0.0067 ← token0 精度提升！
token1: r_max=7.0, r_min=0.2 → s₁=6.8/255=0.0267 ← token1 被自己的7.0拖累
```

**逻辑意义：** token0 保护了自己不会被 token1 的 7.0 污染。

---

###### 完整对比（用我们的 4×4 数据）

| 粒度 | 组数 | 组大小 | 受影响最严重的组 | 最精细的组 | 核心逻辑 |
|:----:|:----:|:------:|:----------------:|:----------:|:--------|
| **Per-tensor** | 1 | 16个元素 | 唯一一组，16个0.1~0.9的小值被6.0拖累 | — | 全校服一个码 |
| **Per-channel** | 4 | 每行4个 | 行0: s=0.0231 | 行2: s=0.0020 | 按身高分男女码 |
| **Per-group** | 8 | 每组2个 | 行1组B: s=0.0224 | 行0组A: s=0.0016 | 按身高每5cm一档 |
| **Per-token(激活)** | 2 | 每行4个 | token1: s=0.0267 | token0: s=0.0067 | token不被其他token污染 |

---

这样用 4×4 的矩阵讲，逻辑线索是不是更清晰了？核心就一句话：

> **粒度越细，离群值的"污染半径"越小。**

如果需要，继续往下讲 **"量化对象"**——我们在同样的 W 和 X 上，看 Weight、Activation、KV Cache 各自量化时面临的不同挑战。

## 1.3 模型量化对象

- **Weight（权重）**：权重量化是最常见的量化对象。量化权重可达到减少模型内存占用空间。权重在训练完后固定，数值范围与输入无关，可离线完成量化，通常相对容易量化。
- **Activation（激活）**：实际上激活往往是占内存使用的大头，因此量化激活不仅可以大大减少内存占用。更重要的是，结合权重量化可以充分利用整数计算获得模型推理性能的提升。但激活输出随输入变化而变化，需要统计数据动态范围，通常更难量化。
- **KV Cache**：除了权重和激活量化之外，在大语言模型中的 KV 缓存也会消耗不少的内存。因此，量化 KV 缓存对于提高模型长序列生成的吞吐量至关重要。
- **Gradient（梯度）**：相对上面的量化对象，略微小众一些，主要用于训练场景。在训练深度学习模型时，梯度通常是浮点数，量化梯度可以在分布式计算中减少通信开销，同时，也可以减少反向传播时的开销。

### 用数据看三个对象的本质区别

在 Transformer 层中，计算流程是 `Y = X · Wᵀ`，具体到元素：`Y[i][j] = Σₖ X[i][k] × W[j][k]`。

**① Weight：静态、离线、固定**
- 训练完成后就固定，数值分布已知、离群值位置固定、量化代价一次性的
- 可以部署前离线扫描整个矩阵，提前算好各种粒度的 scale
- 核心挑战：**如何用更少的 bit 存储 W，同时不让输出误差太大**——是"压缩比 vs 精度"的权衡
- 典型方案：GPTQ(W4A16)、AWQ(W4A16)、FP6-LLM(W6A16)

**② Activation：动态、在线、依赖输入**
- 每次推理都在变，离群值位置不确定（这次 token1 的 c1 是 7.0，下次可能 token0 的 c3 是 9.0）
- 量化参数必须在线算，不能在部署前预计算
- **为什么激活比权重难量化**：权重的离群值影响"存储精度"（一次性），激活的离群值影响"每次前向的计算精度"；且激活量化后每个 X[i][k] 都带误差，会传播到输出
- 核心挑战：**每次推理分布不同，离群值动态出现，如何在不引入过大运行时开销的前提下完成量化**
- 典型方案：LLM.int8()（实时检测离群列）、SmoothQuant（转移难度到权重）、ZeroQuant（per-token 动态）、QuaRot（旋转消除离群值）

**③ KV Cache：逐序列累积、self-attention 专用、内存瓶颈**
- 自回归生成中逐 token 累积：`生成第1个token存入K₁,V₁ → 生成第2个token存入K₂,V₂ → ...`
- 大小随序列长度线性增长，是长上下文推理的内存瓶颈
- **特殊挑战 1（逐 token 到达）**：量化时不知道后续 token 的分布，K₃ 来了个 5.0 离群值，之前的 (s,z) 全被污染
- **特殊挑战 2（K 和 V 分布不同）**：Key 某些通道跨 token 变化大（通道性离群值）→ per-channel 更合理；Value 更分散 → per-token 更合理。这就是 KIVI 的设计依据
- **特殊挑战 3（Attention Sink）**：第一个 token（如 `<s>`）对量化误差异常敏感，所有后续 token 都会关注它，误差会被放大 → KVQuant 保留第一个 token 为 FP16

**三者的本质区别（汇总表）**

| 维度 | Weight（权重） | Activation（激活） | KV Cache |
|:----|:-------------|:-----------------|:---------|
| **变还是不变？** | 训练后永远不变 | 每次推理随输入变 | 同一次推理中逐token增长 |
| **何时知道分布？** | 部署前预知 | 推理时才知道 | 只能知道已生成的部分 |
| **离群值特点** | 位置固定，可预判 | 位置动态，不可预判 | Key有通道性离群值，Value没有 |
| **量化的主要目标** | 减少模型存储/加载 | 利用INT8计算加速 | 支持更长序列/更大batch |
| **谁是瓶颈？** | 显存占用 | 计算吞吐 | 显存占用（长序列时） |
| **量化难度** | ⭐⭐ 中等 | ⭐⭐⭐ 最难 | ⭐⭐ 中等（但策略复杂） |
| **典型位宽** | W4~W8 | A4~A8 | KV2~KV8 |

> 🔍 **直观类比**
>
> - **权重量化 = 压缩一本书**：书已写好，可离线慢慢找最佳压缩方案，误差就在纸上不会再变。
> - **激活量化 = 实时翻译一个人的讲话**：永远不知道下一句是"今天天气不错"还是"超导量子干涉仪"，必须在线处理。
> - **KV Cache 量化 = 边听讲座边记笔记**：笔记越记越多纸不够用了，要压缩笔记，但不知道后面教授还会讲什么。

---

### 📌 原始 Q&A：模型量化对象（历史第 14 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 14 轮）

再来详细的讨论下这部分：模型量化对象

##### A（历史第 15 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！继续用我们那组数据，深入看 Weight、Activation、KV Cache 这三个量化对象在逻辑上的本质区别。

---

###### 设定回顾

```
序列长度 T = 2 (token0, token1)
隐藏维度 d = 4
权重 W: 4×4
激活 X: 2×4
```

**权重 W：**
```
out0: [0.5, 0.1, 0.3, 6.0]   ← c3有离群值
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]
```

**激活 X：**
```
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]  ← c1有离群值
```

在 Transformer 层中，计算流程是：

```
Y = X · Wᵀ
    激活 乘 权重

Y(t×d) = X(t×d) · Wᵀ(d×d)
```

具体到元素：

```
Y[i][j] = Σₖ X[i][k] × W[j][k]
         ↑         ↑      ↑
    第i个token  的 第k维  乘以 第j个输出神经元的第k维权重
```

好，现在来看三个量化对象的本质区别。

---

###### 一、Weight（权重量化）

###### 核心特征：静态、离线、固定

权重在 **训练完成后就固定了**，不会再变。这意味着：

1. **数值分布是已知的** — 你可以在部署前离线扫描整个矩阵
2. **离群值的位置是固定的** — 可以提前找到并做特殊处理
3. **量化代价是一次性的** — 花再多时间算 scale 也只做一次

###### 在我们数据上的体现

```
W 是死的：
         c0   c1   c2   c3
out0:  [0.5, 0.1, 0.3, 6.0]    ← 6.0 永远在 (out0, c3) 这个位置
out1:  [0.4, 0.2, 0.6, 0.8]
out2:  [0.7, 0.3, 0.5, 0.2]
out3:  [0.1, 0.9, 0.4, 0.6]
```

**我们可以在部署前** 就提前算出各种粒度的 scale 值，放内存里。推理时直接查。

###### 所以权重量化的核心挑战是

> **如何用更少的 bit 存储 W，同时不让 Y = X · Wᵀ 的输出误差太大？**

不是实时性——而是"压缩比 vs 精度"的权衡。

###### 典型方案

| 方案 | 位宽 | 思路 |
|:----|:----:|:----|
| **GPTQ** | W4A16 | 逐组量化 + 优化视角：量化后调整未量化权重来补偿误差 |
| **AWQ** | W4A16 | 保护显著通道（根据激活分布找），不优化的权重用更细的尺度 |
| **FP6-LLM** | W6A16 | 6bit 浮点，不换精度格式而是换 kernel |

**权重量化的"难"不是因为它动态变化，而是因为位宽越低，压缩掉的信息越不可恢复。**

---

###### 二、Activation（激活量化）

###### 核心特征：动态、在线、依赖输入

激活值 **每输入一组新数据就在变**。这带来了根本性的不同：

1. **数值分布未知** — 没人知道用户会输入什么文本
2. **离群值的位置不确定** — 这次 token1 的 c1 是 7.0，下次可能 token0 的 c3 是 12.0
3. **量化参数必须在线算** — 不能在部署前预计算

###### 在我们数据上的体现

```
推理时 X 长什么样，取决于你送进去什么句子：

第一次推理（"苹果" → token0="苹果", token1="是水果"）：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]  ← c1 意外很大

第二次推理（"太阳" → token0="太阳", token1="从东边升起"）：
token0: [1.5, 0.3, 0.6, 0.2]
token1: [0.5, 0.4, 0.3, 9.0]  ← c3 意外很大
```

**离群值** （outliers） 在 Transformer 的激活中是一个**固有现象**，学术界认为这与 Attention 机制本身有关——某些特征维度在大多数 token 上值很小，但在少数 token 上会变得极大。

###### 进一步看为什么激活比权重难量化

拿我们第一次推理来算 Y = X·Wᵀ：

```
Y[0][0] = X[token0]·W[out0] = 2.0×0.5 + 0.5×0.1 + 0.8×0.3 + 0.3×6.0 = 1.0 + 0.05 + 0.24 + 1.8 = 3.09
Y[1][0] = X[token1]·W[out0] = 0.4×0.5 + 7.0×0.1 + 0.2×0.3 + 0.6×6.0 = 0.2 + 0.7 + 0.06 + 3.6 = 4.56
```

如果 **激活被量化了**（比如 INT8），每一个 X[i][k] 都带着量化误差：

```
X 量化后再反量化：
per-tensor 量化 X：
全局 min=0.2, max=7.0 → s=6.8/255=0.0267

token0: 2.0→round(2.0/0.0267)=75→75×0.0267=2.00 ✅  误差小
        0.5→round(0.5/0.0267)=19→19×0.0267=0.507 ❌  误差 ~1.5%  ← 注意！
        0.3→round(0.3/0.0267)=11→11×0.0267=0.294 ❌
        ...
token1: 7.0→round(7.0/0.0267)=262→clip到127→127×0.0267=3.39 ❌❌ 大截断误差！
```

**一个关键区别：权重的离群值影响** **存储精度**，激活的离群值影响 **每次前向的计算精度**。

###### 所以激活量化的核心挑战是

> **每次推理的激活分布不同，离群值动态出现，如何在不引入过大运行时开销的前提下完成量化？**

###### 典型方案

| 方案 | 做法 |
|:----|:----|
| **LLM.int8()** | 实时检测离群值列，FP16 算离群部分，INT8 算正常部分 |
| **SmoothQuant** | 不直接量化激活，而是用平滑因子把"激活的量化难度"转移到权重上 |
| **ZeroQuant** | per-token 动态量化，并且把量化操作融合到前一个算子中减少开销 |
| **QuaRot** | 用 Hadamard 旋转矩阵消除异常值，让激活变得容易量化 |

---

###### 三、KV Cache 量化

###### 核心特征：逐序列累积、self-attention 专用、内存瓶颈

KV Cache 不是一层网络的输入输出，而是 **self-attention 中存储的 Key 和 Value 矩阵**，用于避免重复计算。在自回归生成中：

```
生成第 1 个 token: 计算 K₁, V₁  → 存入 KV Cache
生成第 2 个 token: 计算 K₂, V₂  → 存入 KV Cache
生成第 3 个 token: 计算 K₃, V₃  → 存入 KV Cache → 此时 KV Cache = [K₁,K₂,K₃; V₁,V₂,V₃]
...
```

假设我们的模型生成序列长度从 2 扩展到 8：

```
序列长度=2: KV Cache = 2×4矩阵   → 8 个值
序列长度=8: KV Cache = 8×4矩阵   → 32 个值
序列长度=4096: KV Cache = 4096×4 = 16384 个值 × 层数 × 头数！
```

**KV Cache 的大小随序列长度线性增长**，是长上下文推理的内存瓶颈。

###### 在我们数据上的体现

模拟自回归生成过程，假设到第3步时 KV Cache 的样子：

```
Key 缓存（3×4）：
        c0   c1   c2   c3
step1: [0.5, 0.2, 0.8, 0.3]
step2: [0.4, 0.6, 0.2, 0.7]
step3: [0.8, 5.0, 0.3, 0.1]   ← step3 的 c1 出现离群值

Value 缓存（3×4）：
        c0   c1   c2   c3
step1: [0.3, 0.7, 0.1, 0.5]
step2: [0.6, 0.2, 0.4, 0.8]
step3: [0.1, 0.4, 0.9, 0.2]
```

###### KV Cache 量化的特殊挑战

**① 逐 token 到达，不能一次性扫描全部**

权重可以一次性离线扫描；激活在每次推理时完整拿到 X 矩阵。但 KV Cache **是逐 token 生成的**，量化时你不知道后续 token 的分布。

```
问题：
step1: 来 K₁ 了 → 我要不要根据 K₁ 确定 (s,z)？
step2: 来 K₂ 了 → 之前定的 (s,z) 还合适吗？
step3: 来 K₃ 了 → 出现 5.0 离群值！之前的 (s,z) 全被污染了！
```

**② Key 和 Value 的分布不同**

```
Key: 某些通道在不同 token 间变化很大（c1在step1=0.2, step3=5.0）
Value: 更分散，没有这么明显的通道性离群值
```

这就是为什么答主原文中写：

> **KIVI 对 Key 缓存按通道量化，对 Value 缓存按 token 量化**

Key 的特征是"某些通道整体偏大/偏小，跨 token 有规律" → per-channel 更合理  
Value 的特征是"每行（每个 token）自己的数值范围" → per-token 更合理

**③ Attention Sink 现象**

答主原文提到 KVQuant 中的 Attention Sink 感知量化：

> **第一个 token 对量化误差异常敏感**

第一个 token 通常是"全局引导"token（如 `<BOS>` 或 `<s>`），它和其他 token 的注意力模式完全不同，量化它时如果丢了精度，整个序列的推理质量都会下降。

**所以 KVQuant 的做法是：**
> 第一个 token 保留 FP16 不量化，后面 token 正常量化

---

###### 三者的本质区别（汇总）

| 维度 | Weight（权重） | Activation（激活） | KV Cache |
|:----|:-------------|:-----------------|:---------|
| **变还是不变？** | 训练后永远不变 | 每次推理随输入变 | 同一次推理中逐token增长 |
| **何时知道分布？** | 部署前预知 | 推理时才知道 | 只能知道已生成的部分 |
| **离群值特点** | 位置固定，可预判 | 位置动态，不可预判 | Key有通道性离群值，Value没有 |
| **量化的主要目标** | 减少模型存储/加载 | 利用INT8计算加速 | 支持更长序列/更大batch |
| **谁是瓶颈？** | 显存占用 | 计算吞吐 | 显存占用（长序列时） |
| **量化难度** | ⭐⭐ 中等 | ⭐⭐⭐ 最难 | ⭐⭐ 中等（但策略复杂） |
| **典型位宽** | W4~W8 | A4~A8 | KV2~KV8 |

---

###### 一个直观的类比

- **权重量化 = 压缩一本书**  
  书已经写好了，你可以离线慢慢找最佳压缩方案。误差就在纸上，不会再变。

- **激活量化 = 实时翻译一个人的讲话**  
  你永远不知道他下一句是"今天天气不错"还是"超导量子干涉仪"——必须在线处理。

- **KV Cache 量化 = 边听讲座边记笔记**  
  笔记越记越多，纸不够用了。你要压缩笔记内容，但不知道后面教授还会讲什么。

---

这三个量化对象你理清了吗？如果清楚了，我们接下来可以继续看 **"静态量化 vs 动态量化"** 或直接选一个具体的量化方法（比如 SmoothQuant 或 AWQ）从原理到计算走一遍？

## 1.4 静态量化与动态量化

通常，**对于激活而言**，静态量化是指如果采用具有代表性的校准数据集来为其生成缩放因子和零点，这些参数在模型的整个生命周期中保持不变。静态量化的优点在于推理时的计算效率较高，因为它不需要在运行时动态计算量化参数。然而，由于量化参数是固定的，静态量化可能会引入一些量化误差，从而影响模型的精度。

而动态量化是指在每次前向传递期间计算激活的最小值和最大值，以提供动态的缩放因子以实现高精度。动态量化的优点在于它可以更准确地表示模型的激活值，因为它考虑了运行时的实际数据分布。然而，这种方法的缺点是可能会增加计算开销，因为需要在运行时计算量化参数。动态量化适合于那些对模型精度要求较高的应用场景，尤其是当模型的输入数据分布变化较大时。

目前，常见的是对激活使用静态量化，其中最小/最大范围是在离线校准阶段计算的。但由于 LLM 中激活范围差异巨大，将导致准确度显著下降。

> 🔍 **先明确：这两个概念是针对谁的？**
>
> 答主原文说得很清楚：**"对于激活而言"**。静态 vs 动态 = **是否使用校准数据集**提前生成激活的量化参数 (s, z)。
>
> **权重量化永远是静态的**（权重不变，当然离线做），所以这两个概念主要是针对激活量化说的。

### 用数据推演静态 vs 动态

两次推理的激活（注意第二次的离群值从 c1 跑到了 c3）：

```
第一次推理（"苹果好吃"）：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]   ← c1 离群 7.0

第二次推理（"太阳很大"）：
token0: [1.5, 0.3, 0.6, 0.2]
token1: [0.5, 0.4, 0.3, 9.0]   ← c3 离群 9.0 ！
```

**静态量化**：校准阶段（部署前）收集校准集统计量，固定 s/z。

```
校准样本1：token0[2.0,0.5,0.8,0.3] token1[0.4,7.0,0.2,0.6]
校准样本2：token0[1.2,0.4,0.6,0.2] token1[0.3,0.5,0.7,0.1]

校准集 r_min=0.1, r_max=7.0
s_static = (7.0-0.1)/255 = 0.0271   ← 固定下来写入模型

推理时直接用：
第一次推理：token1 的 7.0 → 7.0/0.0271=258 → clip 到 127 ← 勉强在边界
第二次推理：token1 的 9.0 → 9.0/0.0271=332 → clip 到 127 ← 被截断了！误差大！
```

**静态量化的核心问题**：校准集覆盖不到的真实推理数据如果出现更大的离群值（9.0 > 校准集的 7.0），就被截断了。风险在于校准集质量决定了量化精度。

**动态量化**：每次前向实时扫描当前激活，算当次的 min/max。

```
第一次推理：r_max=7.0 → s=(7.0-0.2)/255=0.0267
第二次推理：r_max=9.0 → s=(9.0-0.2)/255=0.0345  ← 重新算！
```

**动态量化的核心优势**：每次都用当次的 (s,z)，**永远不会因为离群值超出预定义范围而被截断**。缺点是每次推理都多一步扫描+计算 s/z 的开销。

**对比表**

| | 静态量化 | 动态量化 |
|:--|:--------|:--------|
| (s,z) 来源 | 校准集统计，固定 | 当前输入统计，实时 |
| 对校准集依赖 | 强（质量决定精度） | 无 |
| 对离群值适应性 | 弱（超范围就截断） | 强（动态调整） |
| 推理额外开销 | 无 | 有（扫描+计算开销） |
| 适用场景 | 高吞吐、输入分布稳定 | 精度敏感、输入分布变化大 |

> ⚠️ **易错校正：激活静态量化 ≠ "激活离线量化好"**
>
> 未来请求的激活值在部署前根本不知道，**不可能提前把所有 activation 数值量化好**。静态的其实是 **activation quantizer 的参数（s,z）固定**，而不是"activation tensor 本身已提前生成并量化"。运行时 X 仍是运行时才产生，只是用它套用固定的 s,z。

> 🔍 **各方法的激活量化策略**
>
> | 方法 | 激活量化策略 | 为什么 |
> |:----|:-----------|:-------|
> | **SmoothQuant** | 静态 | 平滑因子把激活难度转移走，激活变"规整"，静态就够了 |
> | **ZeroQuant** | 动态（per-token） | 不做校准，每次推理逐token动态算，用 kernel 融合抵消开销 |
> | **LLM.int8()** | 动态 | 离群值实时检测 + 混合精度，本质是动态策略 |
> | **QuaRot** | 静态 | 旋转矩阵消除了异常值，激活分布规整，静态就够了 |

---

### 📌 原始 Q&A：静态量化与动态量化（历史第 16 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 16 轮）

静态量化与动态量化
 这部分的呢

##### A（历史第 17 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！继续用我们那组 4×4 的数据来推演。

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

激活 X(2×4) — 第一次推理（"苹果好吃"）：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]

激活 X(2×4) — 第二次推理（"太阳很大"）：
token0: [1.5, 0.3, 0.6, 0.2]
token1: [0.5, 0.4, 0.3, 9.0]   ← 离群值从 c1 跑到了 c3 ！
```

---

###### 先明确这两个概念是针对谁的

答主原文说得很清楚：

> **"对于激活而言"**，静态量化与动态量化的区别在于：**是否使用校准数据集** 来提前生成激活的量化参数 (s, z)。

**权重量化永远是静态的**（权重不变，当然离线做），所以这两个概念主要是针对 **激活量化** 说的。

---

###### 静态量化（Static Quantization）

###### 原理

> 拿一批"代表性"的校准数据（比如 512 条文本），跑一遍推理，**收集** 每层激活值的 min/max 统计信息，算出 (s, z)，**固定下来**。推理时直接用，不再重新算。

###### 用我们数据推演

**校准阶段**（部署前，离线做）：

准备校准数据集，假设就 2 条样本：

```
校准样本1（"苹果好吃"）：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]

校准样本2（"天气不错"）：
token0: [1.2, 0.4, 0.6, 0.2]
token1: [0.3, 0.5, 0.7, 0.1]
```

统计所有校准样本中激活值的全局分布：

```
校准集激活值集合 = [2.0, 0.5, 0.8, 0.3, 0.4, 7.0, 0.6,
                   1.2, 0.4, 0.6, 0.2, 0.3, 0.5, 0.7, 0.1]

校准集 r_min = 0.1     校准集 r_max = 7.0
校准集 s_static = (7.0 - 0.1) / 255 = 0.0271
校准集 z_static = 0（对称量化）
```

然后 **把这个 s_static 固定下来**，写入模型文件。

**推理阶段**（线上，直接使用）：

第一次推理（"苹果好吃"）直接用固定好的 (s_static, z_static) 量化激活：

```
token0: [2.0, 0.5, 0.8, 0.3]
  → 2.0/0.0271=73.8→74, 0.5/0.0271=18.5→19, 0.8/0.0271=29.5→30, 0.3/0.0271=11.1→11

token1: [0.4, 7.0, 0.2, 0.6]
  → 0.4/0.0271=14.8→15, 7.0/0.0271=258.3→clip到127, 0.2/0.0271=7.4→7, 0.6/0.0271=22.1→22
  ← token1的7.0被截断了，误差大！
```

第二次推理（"太阳很大"）同样用这个 (s_static, z_static)：

```
token0: [1.5, 0.3, 0.6, 0.2]
  → 1.5/0.0271=55.4→55, 0.3/0.0271=11.1→11, 0.6/0.0271=22.1→22, 0.2/0.0271=7.4→7

token1: [0.5, 0.4, 0.3, 9.0]
  → 0.5/0.0271=18.5→19, 0.4/0.0271=14.8→15, 0.3/0.0271=11.1→11, 9.0/0.0271=332.2→clip到127
  ← token1的9.0也被截断了，误差更大！
```

###### 静态量化的核心问题

> **校准集覆盖不到的真实推理数据，如果出现更大的离群值，就遭了。**

在我们的例子中：校准集 r_max=7.0，但第二次推理时实际来了个 9.0，直接被截断。

**所以静态量化的风险在于：**
- 校准集质量决定了量化精度
- 如果真实部署时的数据分布和校准集不一致，精度会下降

###### 优点

- 推理时**零额外开销** — 不用算 min/max，不用算 s/z，直接用量化整数算
- 适合吞吐量要求高的场景

---

###### 动态量化（Dynamic Quantization）

###### 原理

> 不提前固定 (s, z)。每次前向传播时，**实时** 扫描当前激活值，算 min/max，得到当前这轮推理专用的 (s, z)。

###### 用我们数据推演

**推理阶段**（线上，实时计算）：

第一次推理，拿到 X 后 **即时统计**：

```
X = [2.0, 0.5, 0.8, 0.3, 0.4, 7.0, 0.2, 0.6]
    r_min = 0.2      r_max = 7.0
    s_dynamic = (7.0 - 0.2) / 255 = 0.0267
```

然后按这个 (s_dynamic, z_dynamic) 量化：

```
token0: 2.0→75, 0.5→19, 0.8→30, 0.3→11   ← 都在范围内
token1: 0.4→15, 7.0→255→clip到127, 0.2→7, 0.6→22  ← 7.0 刚好在边界，勉强不截断
```

第二次推理（"太阳很大"），**重新统计**：

```
X = [1.5, 0.3, 0.6, 0.2, 0.5, 0.4, 0.3, 9.0]
    r_min = 0.2      r_max = 9.0
    s_dynamic = (9.0 - 0.2) / 255 = 0.0345
```

按新的 (s_dynamic=0.0345) 量化：

```
token0: 1.5→43, 0.3→9, 0.6→17, 0.2→6
token1: 0.5→14, 0.4→12, 0.3→9, 9.0→255→clip到127
```

注意：**动态量化虽然比静态好（9.0 这次还能在量化范围内），但步长被 9.0 拉到了 0.0345，其他小值的精度变差了**。

token0 的 0.2 → round(0.2/0.0345)=6 → 反量化 6×0.0345=0.207，误差~3.5%

###### 动态量化的核心优势

> **每次推理都用自己的 (s,z)，不会出现"校准集没见过这么大的值"的情况。永远不会因为离群值超出预定义范围而被截断。**

###### 缺点

- **每次推理都要多一步**：扫描激活 → 算 min/max → 算 s/z。这个开销对延迟敏感的应用不可忽视
- 答主原文说：**"可能会增加计算开销，因为需要在运行时计算量化参数"**

---

###### 一张图对比

```
静态量化：
  校准集 ──→ 统计出 (s,z) ──→ 固定下来 ──→ 所有推理都直接用
              (离线，一次)                    (线上，零额外开销)
  
  风险：校准集 r_max=7.0，线上来了 9.0 → 截断误差

动态量化：
  每次推理 ──→ 扫当前 X ──→ 算出本次 (s,z) ──→ 用这个 (s,z) 量化
              (线上，每次都有开销)
  
  好处：r_max 永远匹配本次输入，不会意外截断
```

---

###### 更深一层：为什么答主特别说"对于激活而言"？

因为权重是 **死的**，不存在"动态"一说：

```
权重永久固定：
out0: [0.5, 0.1, 0.3, 6.0]  ← 6.0 永远在这
out1: [0.4, 0.2, 0.6, 0.8]
...

权重的 (s,z) 永远是离线算好的 → 静态
```

但激活是 **活的**，所以才有静态（提前猜分布）vs 动态（到了现场再量）的选择。

---

###### 动态量化的两种子变体（答主原文提到的）

> **在线动态量化**：权重离线静态量化 + 激活运行时动态量化。这是最常见的动态量化的含义。

> **离线动态量化**：权重运行前先量化，激活在运行时动态量化。听起来一样？

区别是："离线动态"这个提法更多强调 **权重量化和激活量化的时机不同** —— 权重是离线做的，激活是动态做的。而"在线动态"是更常见的说法，指的就是激活用动态量化。

---

###### 关键一句话总结

| | 静态量化 | 动态量化 |
|:--|:--------|:--------|
| (s,z) 来源 | 校准集统计，固定 | 当前输入统计，实时 |
| 对校准集依赖 | 强（质量决定精度） | 无 |
| 对离群值适应性 | 弱（超范围就截断） | 强（动态调整） |
| 推理额外开销 | 无 | 有（扫描+计算开销） |
| 适用场景 | 高吞吐、输入分布稳定 | 精度敏感、输入分布变化大 |

---

我们用这个框架再看答主原文中提到的一些方法，它们对激活的量化策略就很清楚了：

| 方法 | 激活量化策略 | 为什么 |
|:----|:-----------|:-------|
| **SmoothQuant** | 静态 | 用了平滑因子把激活的量化难度转移走，激活变得"规整"了，静态就够了 |
| **ZeroQuant** | 动态（per-token） | 不做校准，每次推理逐token动态算，精度好，但用kernel融合抵消开销 |
| **LLM.int8()** | 动态 | 离群值实时检测 + 混合精度，本质上是一种动态策略 |
| **QuaRot** | 静态 | 用旋转矩阵消除了异常值，激活分布规整了，静态就够了 |

需要继续看 **"离线量化与在线量化"** 或 **"线性量化与非线性量化"** 吗？

## 1.5 离线量化与在线量化

离线量化是指模型上线前进行量化并生成缩放因子，而在线量化是指模型运行时进行量化。

**动态与静态量化的区别在于是否使用校准集，而离线与在线量化的区别则是量化的时机不同。**

简单理解：

- **离线静态量化**：在模型上线推理前使用校准集生成缩放因子，对权重和激活进行量化。
- **在线动态量化**：在模型上线推理时，在每次前向传播过程中实时生成缩放因子，对权重和激活进行量化。
- **离线动态量化**：通常是指对权重在运行前先进行量化，对激活在运行时进行动态量化。

> 🔍 **学习追问：离线/在线 = 是否使用校准集？**
>
> **最终确认：**
>
> 1. **离线 vs 在线 = 量化的时机**（上线前 vs 运行时），**不是**是否用校准集。是否用校准集是**静态 vs 动态**的区分。
> 2. **权重 W**：训练后固定，几乎都是**离线**量化好的（上线前就量化完存起来）。
> 3. **激活 A**：可以离线（用校准集提前算好 s/z，即离线静态），也可以在线（运行时实时算，即在线动态）。
> 4. 实际中最常见的组合是：**W 离线量化 + A 离线静态量化（校准集）**，或者 **W 离线量化 + A 在线动态量化**。

**两个正交维度的组合表**

```
                    ┌──────────────┬──────────────┐
                    │   静态        │    动态       │
                    │  (用校准集)   │  (不用校准集)  │
┌─────────┬─────────┼──────────────┼──────────────┤
│ 离线    │ 权重    │ ✅ 永远离线  │  ❌ 不存在    │
│ (上线前)│ 激活    │ ✅ 离线静态  │  ❌ 不存在    │
│         │         │ (SmoothQuant)│              │
├─────────┼─────────┼──────────────┼──────────────┤
│ 在线    │ 权重    │ ❌ 不存在    │  ❌ 不存在    │
│ (运行时)│ 激活    │ ❌ 不存在    │  ✅ 在线动态  │
│         │         │              │  (ZeroQuant)  │
└─────────┴─────────┴──────────────┴──────────────┘
```

> ⚠️ **易错校正：最实用的理解是拆成两件事**
>
> 不要机械地"激活离线 or 在线二选一"，而是拆成：
> 1. **qparams 什么时候确定？**（离线校准 or 在线统计）
> 2. **当前 activation 什么时候真正被 quantize？**（运行时）
>
> 主流 LLM 推理：
> ```
> Weight：通常部署前离线量化
> Activation static：离线校准 qparams + 在线真正量化当前 activation
> Activation dynamic：在线统计 qparams + 在线量化 activation
> ```

> **本质一句话**：激活的 (s,z) 你是"猜好了"（静态/离线，用校准集）还是"到了现场再量"（动态/在线，不用校准集）？

---

### 📌 原始 Q&A：离线量化与在线量化（历史第 18-20 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 18 轮）

所以关于离线和在线的区别就是 是否使用校准集？ w的话基本都是离线的。然后激活的话分离线和在线？

##### A（历史第 19 轮，正文原样保留；仅将内部标题降级以保持目录层级）

你的理解 **基本对，但差一点点**。让我用答主原文的原文来帮你厘清：

---

###### 答主原文

> **动态与静态量化的区别在于是否使用校准集**，而**离线与在线量化的区别则是量化的时机不同**。

所以这是 **两个正交的维度**，不是一回事。

---

###### 两个维度拆开看

###### 维度 A：静态 vs 动态 — 校准集用不用？

| | 静态 | 动态 |
|:--|:----|:----|
| 用不用校准集？ | **用**，提前统计分布，固定 (s,z) | **不用**，运行时根据实际输入算 (s,z) |
| 回答的问题是 | (s,z) **怎么来的**？ | |

###### 维度 B：离线 vs 在线 — 什么时候量化？

| | 离线 | 在线 |
|:--|:----|:----|
| 什么时候量化？ | **上线前/推理前** 量化好 | **运行时** 边推理边量化 |
| 回答的问题是 | 量化这个动作 **什么时候执行**？ | |

---

###### 组合起来是 3 种（答主原文）

答主原文给出了三种组合，**不是四种**，因为有些组合在逻辑上不存在：

###### ① 离线静态量化

> 模型上线推理前使用校准集生成缩放因子，对权重和激活进行量化

```
上线前（离线）：
  校准集 → 统计出 (s,z) → 对 W 和 X 都量化好 → 上线推理直接用
  ↑ 静态                      ↑ 离线
```

**典型**：SmoothQuant、GPTQ、AWQ

###### ② 在线动态量化

> 模型上线推理时，在每次前向传播过程中实时生成缩放因子

```
推理时（在线）：
  拿到当前 X → 实时算 (s,z) → 实时量化激活 → 算完丢掉
  ↑ 动态          ↑ 在线
```

**典型**：ZeroQuant 的激活量化、LLM.int8() 的离群值检测

###### ③ 离线动态量化

> 对权重在运行前先进行量化，对激活在运行时进行动态量化

```
上线前：权重 W 先量化好（离线）
推理时：激活 X 动态算 (s,z)（动态/在线）
```

**典型**：大部分 W4A16 方案 + 激活动态量化的混合方案

---

###### 回到你的问题

> "离线和在线的区别就是是否使用校准集？"

**不是。** 使用校准集是 **静态 vs 动态** 的区别，不是离线 vs 在线。

| 你说的是 | 实际对应 |
|:---------|:---------|
| 是否使用校准集 | **静态 vs 动态** 的区别 |
| 量化时机（上线前 vs 运行时） | **离线 vs 在线** 的区别 |

---

> "W的话基本都是离线的"

**对 ✅**。权重是死的，部署前量化好，永远离线。

> "激活的话分离线和在线？"

**对 ✅**。激活可以选择：
- **离线**（用校准集提前算好 (s,z)，推理时直接用）→ 离线静态
- **在线**（推理时实时算 (s,z)）→ 在线动态，或者叫离线动态（权重离线+激活在线）

---

###### 一个表格帮你记住

```
                    ┌──────────────┬──────────────┐
                    │   静态        │    动态       │
                    │  (用校准集)   │  (不用校准集)  │
┌─────────┬─────────┼──────────────┼──────────────┤
│ 离线    │ 权重    │ ✅ 永远离线  │  ❌ 不存在    │
│ (上线前)│ 激活    │ ✅ 离线静态  │  ❌ 不存在    │
│         │         │ (SmoothQuant)│              │
├─────────┼─────────┼──────────────┼──────────────┤
│ 在线    │ 权重    │ ❌ 不存在    │  ❌ 不存在    │
│ (运行时)│ 激活    │ ❌ 不存在    │  ✅ 在线动态  │
│         │         │              │  (ZeroQuant)  │
└─────────┴─────────┴──────────────┴──────────────┘
```

注意：**离线动态** 其实是"权重离线 + 激活在线动态"的混合体，不是单纯的一个格子。

---

这样清楚了吗？如果清楚了，我们继续往下看 **"线性量化与非线性量化"**，还是直接挑一个具体的量化方法（比如 GPTQ 或 AWQ）上手推？

##### Q（历史第 20 轮）

所以其实就是两种情况对吧，权重 离线做好。然后激活再分是 离线做好用校准集还是在线自己量化？

##### A（历史第 21 轮，正文原样保留；仅将内部标题降级以保持目录层级）

**完全正确！** 就是这个意思。你把两个维度合并成了一个更直观的理解：

---

**第 1 种 — 激活也离线（离线静态量化）**

```
权重 ─── 离线做好 ✓
激活 ─── 用校准集离线算好 (s,z)，固定下来，推理直接用
```

**第 2 种 — 激活在线（在线动态量化 / 离线动态量化）**

```
权重 ─── 离线做好 ✓
激活 ─── 不依赖校准集，推理时拿到真实输入再临时算 (s,z)
```

**本质上就是在问：激活的 (s,z) 你是"猜好了"还是"到了现场再量"？**

- 猜好了 → 用校准集（静态/离线）
- 现场量 → 不用校准集（动态/在线）

这个理解完全正确，简洁明了。要接着往下看 **"线性量化与非线性量化"** 吗？

## 1.6 线性量化与非线性量化

根据量化数据表示的原始数据范围是否均匀，还可以将量化方法分为线性量化和[非线性量化](https://zhida.zhihu.com/search?content_id=710333991&content_type=Answer&match_order=1&q=%E9%9D%9E%E7%BA%BF%E6%80%A7%E9%87%8F%E5%8C%96&zhida_source=entity)。实际的深度神经网络的权重和激活值通常是不均匀的；因此，理论上使用非线性量化导致的精度损失更小，但在实际推理中非线性量化的计算复杂度较高，通常使用线性量化。

线性量化公式：

```
q = clip(round(r / s) + z, q_min, q_max)
```

其中：
- `round(·)` 和 `clip(·)` 分别表示取整和截断操作
- `q_min`、`q_max` 是量化后的最小值和最大值
- `s` 为缩放系数：`s = (r_max - r_min) / (q_max - q_min)`
- `z` 是表示数据偏移的偏置

**对称量化**（z=0）：可以避免量化算子在推理中计算 z 相关的部分，降低推理时的计算复杂度。

**非对称量化**（z≠0）：可以根据实际数据的分布确定最小值和最大值，更加充分地利用量化数据信息，使得量化导致的损失更低。

> 🔍 **学习追问：现实中大家都使用线性量化对吧？**
>
> **是的（绝大多数场景）。** 虽然理论上非线性量化（如 log 量化、codebook 量化）精度损失更小，但硬件（GPU Tensor Core）对线性 INT8/INT4 的支持非常成熟，非线性量化的反量化和计算开销大，难以高效部署。因此工业界几乎全部使用线性量化。
>
> 例外：NF4（QLoRA 使用）是一种非线性的 4bit 数据类型，专门针对正态分布数据优化，但它主要用于微调场景，推理部署仍以线性 INT4 为主。

> ⚠️ **易错校正（2025 视角）**：不能简单理解成"实际部署几乎全是线性 INT"。硬件原生 FP8/FP4 使非均匀浮点量化也可以非常高效，因此应把"算法精度"和"硬件原生支持"一起考虑。

### 用数据推演线性 vs 非线性

以 out0 行 [0.5, 0.1, 0.3, 6.0] 用 INT2（4 个格子）量化为例：

**线性量化（格子均匀）**：步长 = (6.0-0.1)/3 ≈ 1.97，格子落在 [0.1, 2.1, 4.0, 6.0]

```
0.5 → 离格1(0.1)和格2(2.1)都近 → 就近分配 → 反量化 0.1  ❌ 误差 0.4
0.3 → 格1 → 反量化 0.1  ❌ 误差 0.2
6.0 → 格4 → 反量化 6.0  ✅
```

**核心矛盾**：值密集区域 [0.1, 0.5] 只占整个区间 [0.1, 6.0] 的 6.8%，4 个均匀格子分到这里不到 1 个格子，精度严重浪费。

**非线性量化（格子不均匀）**：例如设计格子 [0.1, 0.2, 0.4, 6.0]

```
0.5 → 离格3(0.4)最近 → 反量化 0.4  ❌ 误差 0.1（对比线性 0.4）
0.3 → 离格2(0.2)/格3(0.4)都近 → 反量化 0.2~0.4  ❌ 误差 0.1（对比线性 0.2）
6.0 → 格4 → 反量化 6.0  ✅
```

**非线性把密集区误差从 0.4 降到 0.1**，因为密集区分配了更多格子。

**常见实现方式**：对数量化（log 域映射）、查找表 LUT（如 NF4 的 16 个格子，中间密两边疏，匹配正态分布）。

**各方案线性/非线性选择**：

| 方案 | 线性还是非线性 | 为什么这么选 |
|:----|:-------------|:-----------|
| **GPTQ**：INT4 | 线性 | 为了用 INT4 GEMM 加速 |
| **AWQ**：INT4 | 线性 | 同上 |
| **SmoothQuant**：INT8 | 线性 | 为了用 INT8 Tensor Core |
| **QLoRA**：NF4 | **非线性** | 精度优先，NF4 匹配正态分布 |
| **any4**：LUT | **非线性** | 每行学一个专属 LUT，精度最高 |
| **FP8** | 非线性（浮点本身） | 浮点是指数位带来的非均匀间距，但硬件原生支持 |

> **一句话类比**：线性量化 = 拿一把等距的梳子梳头发，便宜但浓密处梳不干净；非线性量化 = 浓密处齿密、稀疏处齿疏，梳得干净但贵。所以硬件喜欢线性，精度损失通过其他手段（分组、平滑、优化）补偿。

---

### 📌 原始 Q&A：线性量化与非线性量化（历史第 22-24 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 22 轮）

线性量化与非线性量化 这部分呢？

##### A（历史第 23 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好，继续用我们的 4×4 数据来推演！

---

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]
```

---

###### 核心问题：量化格子怎么排？

量化本质上就是：**在实数轴上，用有限的几个离散值（格子）来代表所有的值。**

线性 vs 非线性，问的是：

> **这些格子之间的间距是均匀的，还是不均匀的？**

---

###### 一、线性量化（Linear Quantization）

###### 格子间距相等

```
在实数轴上，格子是均匀排列的：

  0.1       1.6       3.1       4.6       6.0
  |----|----|----|----|----|----|----|----|
  格1  格2  格3  格4  格5  格6  格7  格8
  ↑                              ↑
  s=0.74                         s=0.74
  每个格子宽度都是 0.74
```

用我们之前算过的公式：

```
s = (r_max - r_min) / (q_max - q_min)
  = (6.0 - 0.1) / (255) = 0.0231    ← 每个格子等宽
```

###### 用我们数据看线性量化的问题

把 W 中 out0 行的 4 个值拿出来，用 INT2 量化（2bit，只有 4 个格子，方便演示）：

```
线性量化（4个格子均匀排列）：
  
  0.1           2.1           4.0           6.0
  |──────┬──────┬──────┬──────|
 格1(s=0) 格2(s=1) 格3(s=2) 格4(s=3)
   步长 = (6.0-0.1)/3 ≈ 1.97

out0 行 = [0.5, 0.1, 0.3, 6.0]
  0.5 → 离格1（0.1）和格2（2.1）都近 → 就近分配到格1  → 反量化得 0.1  ❌ 误差 0.4
  0.1 → 格1 → 反量化得 0.1  ✅
  0.3 → 格1 → 反量化得 0.1  ❌ 误差 0.2
  6.0 → 格4 → 反量化得 6.0  ✅
```

**关键问题**：W 中大部分值（0.1, 0.3, 0.5）集中在 [0.1, 0.5] 这个小区间内，但线性量化的格子均匀分布在 [0.1, 6.0] 整个大区间上。**所以 [0.1, 0.5] 这个密集区只分到了不到 1 个格子，精度严重浪费。**

```
整个区间 [0.1, 6.0] 宽度 = 5.9
[0.1, 0.5] 宽度 = 0.4，只占整个区间的 6.8%
4 个格子均匀分布，[0.1, 0.5] 只分到 4 × 6.8% ≈ 0.27 个格子 ← 不够一个格子！
```

**这就是线性量化的核心矛盾：** 值密集的区域没有获得更多的格子，值稀疏的区域浪费了格子。

---

###### 二、非线性量化（Non-linear Quantization）

###### 格子间距不相等

```
值密集的区域（0.1~0.5）格子密，值稀疏的区域（5.0~6.0）格子疏：

  0.1 0.15 0.25 0.4 0.6 1.0 1.8 3.2 5.0 6.0
  |--|----|---|----|---|----|---|----|---|
  格1 格2  格3 格4  格5 格6  格7 格8  格9
  ↑                     ↑
  小步长                 大步长
```

###### 非线性量化的常见实现方式

**方式一：动态/对数量化**

直接用 log 域代替线性域：

```
线性量化的映射： q = round((r - r_min) / s)
对数量化的映射： q = round(log(r) / s_log)
```

**方式二：查找表（LUT）**

不依赖公式，直接定义一个映射表。比如 NF4（QLoRA 用的）：

```
NF4 的 16 个格子（4bit）：
  [-1.0, -0.696, -0.525, -0.393, -0.277, -0.174, -0.085, -0.008,
    0.008, 0.085, 0.174, 0.277, 0.393, 0.525, 0.696, 1.0]
```

看这些格子的间距：
```
-0.008 到 0.008：间距 0.016  ← 很密（值密集区）
-0.696 到 -1.0：间距 0.304  ← 很疏（值稀疏区）
```

**NF4 是为正态分布数据设计的**——因为神经网络权重和激活的分布通常中间密、两边稀，中间用细格子，两边用粗格子，正好匹配。

###### 用我们数据看非线性量化的效果

还是 out0 行 [0.5, 0.1, 0.3, 6.0]，用 INT2（4个格子）非线性量化：

```
假设我们设计的非线性格子（根据数据分布定制）：
  [0.1, 0.2, 0.4, 6.0]
  格子之间间距分别为：0.1, 0.2, 5.6

out0 行 = [0.5, 0.1, 0.3, 6.0]
  0.5 → 离格3（0.4）最近 → 格3 → 反量化得 0.4  ❌ 误差 0.1
  0.1 → 格1 → 反量化得 0.1  ✅
  0.3 → 格3（0.4） vs 格2（0.2）→ 离格2（0.2）更近？不对，|0.3-0.2|=0.1, |0.3-0.4|=0.1 → 选格2或格3都行
  6.0 → 格4 → 反量化得 6.0  ✅
```

对比线性量化的效果（之前算的）：
```
线性：0.5→0.1（误差0.4）  0.3→0.1（误差0.2）
非线性：0.5→0.4（误差0.1） 0.3→0.2（误差0.1）
```

**非线性把密集区 [0.1, 0.5] 的误差从 0.4 降到了 0.1，因为这里分了更多格子。**

---

###### 三、直接对比（用我们数据）

| | 线性量化 | 非线性量化 |
|:--|:--------|:----------|
| **格子怎么排？** | 均匀等距 | 不均匀，密集区细、稀疏区粗 |
| **0.5→多少？** | 0.1（误差0.4） | 0.4（误差0.1） |
| **0.3→多少？** | 0.1（误差0.2） | 0.2（误差0.1） |
| **6.0→多少？** | 6.0（误差0） | 6.0（误差0） |
| **总误差** | 0.6 | 0.2 |
| **优点** | 计算简单，硬件支持好 | 精度高，匹配数据分布 |
| **缺点** | 数据分布不均匀时精度差 | 计算复杂，硬件支持差 |

---

###### 四、答主原文的结论

> **"实际的深度神经网络的权重和激活值通常是不均匀的；因此，理论上使用非线性量化导致的精度损失更小，但在实际推理中非线性量化的计算复杂度较高，通常使用线性量化。"**

**举个例子解释这句话：**

线性量化做矩阵乘法时：

```
Y = X · Wᵀ

如果 X 和 W 都是线性 INT8：
  Y_int = INT8_GEMM(X_int8, W_int8)  ← 硬件原生指令，一条搞定
  Y_fp = Y_int × s_x × s_w            ← 最后反量化一次
```

非线性量化做矩阵乘法时：

```
如果 X 和 W 是非线性量化：
  每个值都要先通过 LUT 查表反量化到 FP16
  然后做 FP16 的矩阵乘法  ← 慢！
```

---

###### 五、实际中的妥协

答主原文中提到的方案，它们的选择：

| 方案 | 线性还是非线性 | 为什么这么选 |
|:----|:-------------|:-----------|
| **GPTQ**：INT4 | 线性 | 为了用 INT4 GEMM 加速 |
| **AWQ**：INT4 | 线性 | 同上 |
| **SmoothQuant**：INT8 | 线性 | 为了用 INT8 Tensor Core |
| **QLoRA**：NF4 | **非线性** | 精度优先，NF4 匹配正态分布，效果好 |
| **any4**：LUT | **非线性** | 每行学一个专属 LUT，精度最高 |
| **FP8** | 非线性（浮点本身） | 浮点就是非线性的，指数位带来非均匀间距 |

---

用一个简单类比收尾：

- **线性量化** = 拿一把等距的梳子梳头发。头发浓密的地方每根梳齿穿过好几根头发，梳不干净，但梳子便宜。
- **非线性量化** = 头发浓密的地方梳齿密，稀疏的地方梳齿疏。梳得干净，但梳子定做起来贵。

所以答主说"通常使用线性量化"——因为硬件喜欢它，**精度损失通过其他手段（分组、平滑、优化）来补偿**。

---

要接着看 **"量化数据类型"** 还是直接挑一个具体的量化方法跑一遍完整流程？

##### Q（历史第 24 轮）

所以基本现实中大家还是使用的是线性的量化对吧？

##### A（历史第 25 轮，正文原样保留；仅将内部标题降级以保持目录层级）

**对，绝大多数场景下用的都是线性量化。**

看几个实际数据：

| 方案 | 类型 | 为什么用线性 |
|:----|:----|:-----------|
| **GPTQ** (W4A16) | 线性 INT4 | GPTQ 是业界部署最广的 PTQ 方案之一 |
| **AWQ** (W4A16) | 线性 INT4 | 被 TensorRT-LLM、vLLM、llama.cpp 等主流框架内置 |
| **SmoothQuant** (W8A8) | 线性 INT8 | NVIDIA TensorRT 官方推荐 |
| **LLM.int8()** | 线性 INT8 | bitsandbytes 库在用 |
| **TensorRT-LLM FP8** | **非线性**（浮点本身） | 但 FP8 有硬件原生支持，所以"好用" |

**唯一大规模部署的非线性方案是 QLoRA 的 NF4**，但 QLoRA 是微调场景，不是推理部署场景。微调时精度比速度重要，NF4 能在 4bit 下比 INT4 保留更多信息，所以值。

推理部署的话，**INT4 AWQ/GPTQ** 和 **FP8** 基本占了绝大多数，它们都是线性量化（FP8 本质上也是线性的分段近似）。

## 1.7 量化数据类型

LLM 主要有三种类型量化：

- **仅权重量化**：只量化每个线性层的权重张量 W。
- **权重激活量化**：量化每个线性层的输入激活 X 和权重张量 W。
- **KV 缓存量化**：量化每个自注意力块中的键张量 K 和值张量 V。

> 🔍 **学习追问：W4A16 等记号是什么意思？**
>
> `W4A16` → 权重(Weight)用 4bit 存，激活(Activation)用 16bit 存。**注意**：这里的数字是"存的时候用多少 bit"，不是"计算的时候用多少 bit"。例如 W4A16，权重虽然用 4bit 存，但计算时通常会反量化回 FP16 再做矩阵乘。

### 仅权重量化

| 类型 | 代表方法 |
|---|---|
| W8A16 | MinMax |
| W6A16 | FP6-LLM |
| W4A16 | AWQ、GPTQ、SpQR、OmniQuant、QuIP# |
| W3A16 | GPTQ、SpQR、OmniQuant、QuIP# |
| W2A16 | OmniQuant、QuIP、QuIP# |

### 权重激活量化

| 类型 | 代表方法 |
|---|---|
| W8A8 | LLM.int8()、SmoothQuant、ZeroQuant |
| W6A6 | OmniQuant |
| W4A8 | QoQ |
| W4A4 | Atom、QuaRot、OmniQuant |

### KV Cache 量化

| 类型 | 代表方法 |
|---|---|
| KV8 | INT8（LMDeploy、TensorRT-LLM）、FP8（TensorRT-LLM、vLLM） |
| KV4 | Atom、QuaRot、QoQ |
| KV3 | KVQuant |
| KV2 | KVQuant、KIVI |

### 三类量化的递进逻辑

> 🔍 答主原文的分类逻辑其实是：
>
> 1. **仅权重量化**（W4A16, W6A16...）：解决"模型太大放不下" → **省显存**；代价是计算时需反量化回 FP16，计算速度不提升甚至略降。
> 2. **权重激活量化**（W8A8, W4A8, W4A4...）：解决"推理速度不够快" → **省计算**；代价是激活量化难度大，精度损失风险高。
> 3. **KV Cache 量化**（KV8, KV4, KV2...）：解决"长序列放不下" → **省 KV 显存**；代价是需要处理逐 token 到达、Attention Sink 等特殊挑战。

### 三种类型的计算流程与权衡对比

| 数据类型 | 权重存储 | 激活存储 | 计算精度 | 省显存 | 省计算 | 典型方法 |
|:--------|:--------:|:--------:|:--------:|:-----:|:-----:|:--------|
| **W8A16** | INT8(1/2) | FP16 | FP16 | 中 | 低 | MinMax |
| **W6A16** | FP6(3/8) | FP16 | FP16 | 中 | 低 | FP6-LLM |
| **W4A16** | INT4(1/4) | FP16 | FP16 | **高** | 低 | AWQ, GPTQ |
| **W8A8** | INT8(1/2) | INT8(1/2) | INT8 | 中 | **高** | SmoothQuant |
| **W4A8** | INT4(1/4) | INT8(1/2) | INT8 | **高** | **高** | QoQ |
| **W4A4** | INT4(1/4) | INT4(1/4) | INT4 | **高** | **高** | Atom, QuaRot |
| **KV8** | — | — | INT8 | 省KV内存 | — | TensorRT-LLM |
| **KV4** | — | — | INT4 | **省更多** | — | Atom |
| **KV2** | — | — | INT2 | **最省** | — | KIVI, KVQuant |

**计算流程对比：**

```
仅权重量化（W4A16）：
  内存:  W_int4(4×4) + scale       ← 1/4 空间
  计算:  W_int4 → 反量化 → W_fp16
         Y = FP16_GEMM(X_fp16, W_fp16)  ← FP16 计算，速度一般
  典型:  AWQ, GPTQ

权重激活量化（W8A8）：
  内存:  W_int8(4×4) + X_int8(2×4)  ← 1/2 空间
  计算:  Y = INT8_GEMM(X_int8, W_int8ᵀ) ← INT8 计算，最快
  典型:  SmoothQuant, LLM.int8()

权重激活量化（W4A8）：
  内存:  W_int4(4×4) + X_int8(2×4)  ← 权重 1/4, 激活 1/2
  计算:  W_int4 → 反量化 → W_int8(中间态)
         Y = INT8_GEMM(X_int8, W_int8ᵀ) ← INT8 计算，快
  典型:  QoQ

KV Cache 量化（KV4）：
  内存:  K_int4 + V_int4            ← 1/4 空间，省显存放长序列
  计算:  Attention 计算时反量化回 FP16
  典型:  Atom, QuaRot, QoQ
```

> ⚠️ **易错校正：W4A16 不只是"存储格式"**
>
> 一个常见误区是把它理解成"只是存储格式"。更准确地说：它首先描述 operand 的量化精度；真正计算时如何使用这些 operand 取决于 kernel。
>
> 例如 W4A16 实际流程可能是：
> ```
> HBM: packed INT4 weight
>   ↓ 加载到片上
> unpack + scale
>   ↓ 可能转换为 FP16/BF16 fragment
> Tensor Core MMA
> ```
>
> kernel 在消费权重的过程中完成解包/反量化，尽量让中间结果只存在寄存器/shared memory 中，而**不是**完整 materialize 一份 FP16 W。

---

### 📌 原始 Q&A：量化数据类型（历史第 28 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 28 轮）


---
我们继续讨论 量化数据类型
 这部分的

##### A（历史第 29 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！继续用我们那组 4×4 的数据来推演 **量化数据类型** 这部分。

---

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

激活 X(2×4)：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]
```

---

###### W 和 A 分别代表什么

```
W4A16  →  权重(Weight)用 4bit 存，激活(Activation)用 16bit 存
W8A8   →  权重用 8bit，激活也用 8bit
W4A8   →  权重用 4bit，激活用 8bit
```

**注意**：这里的数字是"存的时候用多少 bit"，不是"计算的时候用多少 bit"。比如 W4A16，权重虽然用 4bit 存，但计算时通常会反量化回 FP16 再做矩阵乘。

---

###### 答主原文的分类框架

LLM主要有三种类型量化：

| 类型 | 量化对象 | 典型位宽 |
|:----|:---------|:--------|
| **仅权重量化** | 只量化权重 W | W8A16, W6A16, W4A16, W3A16, W2A16 |
| **权重激活量化** | 同时量化权重 W 和激活 X | W8A8, W6A6, W4A8, W4A4 |
| **KV缓存量化** | 量化 Key 和 Value | KV8, KV4, KV3, KV2 |

这三种类型对应三种不同的 **优化目标**，我们逐个看。

---

###### 一、仅权重量化（Weight-only Quantization）

###### 核心目标：省显存

> 模型太大，GPU 放不下，所以把权重压缩了存，计算时再解压。

用我们 4×4 的权重 W，看不同位宽占多少空间：

```
W 原始（FP16）：  4×4 × 16bit = 256 bit = 32 字节
W4（INT4）：      4×4 × 4bit  = 64 bit  = 8 字节
W8（INT8）：      4×4 × 8bit  = 128 bit = 16 字节
W6（FP6）：       4×4 × 6bit  = 96 bit  = 12 字节
```

**压缩比**：W4 是 FP16 的 1/4，W8 是 FP16 的 1/2。

###### W8A16 — MinMax

拿我们 W 的一行来演示 INT8 量化：

```
out0: [0.5, 0.1, 0.3, 6.0]

INT8 对称量化：
s = max(|0.5|, |0.1|, |0.3|, |6.0|) / 127 = 6.0 / 127 ≈ 0.0472

量化：
  0.5 → round(0.5/0.0472) = round(10.6) = 11
  0.1 → round(0.1/0.0472) = round(2.1)  = 2
  0.3 → round(0.3/0.0472) = round(6.4)  = 6
  6.0 → round(6.0/0.0472) = round(127.0) = 127 ✅

反量化：
  11 → 11 × 0.0472 = 0.519  ← 误差 ~3.8%
  2  → 2 × 0.0472 = 0.094   ← 误差 ~6%
  6  → 6 × 0.0472 = 0.283   ← 误差 ~5.6%
  127 → 127 × 0.0472 = 6.0  ← 精确
```

W8 精度还不错，误差在可接受范围。但如果我们想压到 **W4** 呢？

###### W4A16 — AWQ / GPTQ

INT4 只有 16 个格子（-8 ~ 7），out0 行 [0.5, 0.1, 0.3, 6.0] 直接量化：

```
INT4 对称量化：
s = 6.0 / 7 ≈ 0.857

量化：
  0.5 → round(0.5/0.857) = round(0.58) = 1 → 反量化 0.857  ← 误差 0.357 ❌
  0.1 → round(0.1/0.857) = round(0.12) = 0 → 反量化 0       ← 误差 0.1  ❌
  0.3 → round(0.3/0.857) = round(0.35) = 0 → 反量化 0       ← 误差 0.3  ❌
  6.0 → round(6.0/0.857) = round(7.0)  = 7 → 反量化 6.0     ← 精确
```

小值全被"吞"成 0 了！这就是为什么 **W4 不能简单 RTN，需要 GPTQ / AWQ 这类高级方法**。

**但是计算时**：激活是 FP16，所以矩阵乘是：

```
Y = X(W4反量化后) · Wᵀ
  = X(FP16) · W(反量化回FP16)

Y[0][0] = X[token0] · W_recon[out0]
  = 2.0×0.857 + 0.5×0 + 0.8×0 + 0.3×6.0
  = 1.714 + 0 + 0 + 1.8 = 3.514

原始 FP16 结果：3.09
量化后结果：     3.514  ← 误差 ~13.7% ❌
```

这就是 GPTQ/AWQ 要解决的问题——通过优化/保护显著权重来减少这个误差。

**计算流程示意**：

```
内存中：  W_int4[4×4]  +  scale/zero 元数据
计算时：  W_int4 → 反量化 → W_fp16
          Y = X_fp16 · W_fp16ᵀ  ← 计算时用的是 FP16
```

---

###### 二、权重激活量化（Weight-Activation Quantization）

###### 核心目标：算得快

> 权重和激活都是低精度，直接用 INT8/INT4 GEMM 做矩阵乘，不用反量化，计算更快。

###### W8A8 — LLM.int8() / SmoothQuant

权重和激活都用 INT8，矩阵乘可以直接用 INT8 GEMM：

```
W_int8(4×4) · X_int8(2×4) → Y_int32(2×4)

最后再把 Y_int32 反量化回 FP16
```

**关键区别**：W8A8 不需要"先反量化再做矩阵乘"，而是直接做 INT8 矩阵乘。

用 SmoothQuant 平滑后的数据演示（假设平滑因子 s=0.5）：

```
平滑前：  X = [2.0, 0.5, 0.8, 0.3]
          W_out0 = [0.5, 0.1, 0.3, 6.0]

平滑后：  X' = X/s = [4.0, 1.0, 1.6, 0.6]
          W' = W×s = [0.25, 0.05, 0.15, 3.0]

Y[0][0] = 4.0×0.25 + 1.0×0.05 + 1.6×0.15 + 0.6×3.0
        = 1.0 + 0.05 + 0.24 + 1.8 = 3.09 ✅  ← 数学上等价于原始结果
```

###### W4A8 — QoQ

权重 4bit 存，激活 8bit 存。计算时：**权重反量化为 8bit**，然后用 INT8 GEMM 算。

```
内存中：  W_int4[4×4] + scale元数据    ← 省显存
          X_int8[2×4]                   ← 省显存

计算时：  W_int4 → 反量化 → W_int8(中间态)
          Y = INT8_GEMM(X_int8, W_int8ᵀ)
          Y → 反量化 → Y_fp16
```

为什么这么做？答主原文说：

> 4位量化权重张量将首先反量化为中间8位量化权重张量，然后执行INT8矩阵乘法，就好像是 W8A8 逐通道（per-channel）量化一样。

**W4A8 vs W4A16 的区别**：

```
W4A16：W_int4 → 反量化 → W_fp16 → FP16_GEMM(X_fp16, W_fp16)
        计算是 FP16 的，慢

W4A8： W_int4 → 反量化 → W_int8 → INT8_GEMM(X_int8, W_int8)
        计算是 INT8 的，快
```

###### W4A4 — Atom / QuaRot

权重和激活都只有 4bit。计算时直接用 INT4 GEMM（如果能用的话），或者扩展到 INT8 做。

但答主原文也警告了：

> W4A4量化将严重影响LLMs的效果
> W4A4会导致所有模型完全丧失涌现能力

**为什么？** 用我们数据看：激活 4bit 只有 16 个格子。

```
X token0: [2.0, 0.5, 0.8, 0.3]

INT4 对称量化：
s = max(2.0, 0.5, 0.8, 0.3) / 7 = 2.0/7 ≈ 0.286

2.0 → round(2.0/0.286)=7  → 反量化 2.0   ✅
0.5 → round(0.5/0.286)=2  → 反量化 0.572 ❌ 误差 0.072
0.8 → round(0.8/0.286)=3  → 反量化 0.858 ❌ 误差 0.058
0.3 → round(0.3/0.286)=1  → 反量化 0.286 ❌ 误差 0.014

然后这组有误差的激活 × 有误差的权重（也是 INT4）→ 误差乘法放大！
```

**这就是 W4A4 难的核心原因**：权重和激活两边的误差相乘，误差放大。

---

###### 三、KV Cache 量化

###### 核心目标：支持长序列

KV Cache 随序列长度线性增长。答主原文：

> KV Cache 也会消耗不少的内存

假设我们模型的序列长度从 2 扩展到 100：

```
KV Cache FP16：每个token存 K + V = 2 × 4 × 16bit = 128 bit = 16 字节
100 个token：100 × 16 = 1600 字节

KV Cache INT4：每个token存 K + V = 2 × 4 × 4bit = 32 bit = 4 字节
100 个token：100 × 4 = 400 字节 ← 省了 4 倍！
```

配对组合参考答主原文：

| 文本长度 | 推荐 |
|:--------|:----|
| 短文本 | W8KV4 几乎无损失 |
| 长文本 | W8KV8 是更好的选择 |

---

###### 四、三种数据类型的三维对比

用我们的数据，看同一个推理 Y = X·Wᵀ 在不同数据类型下的行为：

###### 计算流程对比

```
仅权重量化（W4A16）：
  内存:  W_int4(4×4)  +  scale        ← 1/4 空间
  计算:  W_int4 → 反量化 → W_fp16
         Y = FP16_GEMM(X_fp16, W_fp16)  ← FP16 计算，速度一般
  典型:  AWQ, GPTQ

权重激活量化（W8A8）：
  内存:  W_int8(4×4)  +  X_int8(2×4)   ← 1/2 空间
  计算:  Y = INT8_GEMM(X_int8, W_int8ᵀ) ← INT8 计算，最快的
  典型:  SmoothQuant, LLM.int8()

权重激活量化（W4A8）：
  内存:  W_int4(4×4)  +  X_int8(2×4)   ← 权重 1/4, 激活 1/2
  计算:  W_int4 → 反量化 → W_int8
         Y = INT8_GEMM(X_int8, W_int8ᵀ) ← INT8 计算，快
  典型:  QoQ

KV Cache 量化（KV4）：
  内存:  K_int4 + V_int4                ← 1/4 空间，省显存放长序列
  计算:  Attention 计算时反量化回 FP16
  典型:  Atom, QuaRot, QoQ
```

###### 一个表格

| 数据类型 | 权重存储 | 激活存储 | 计算精度 | 省显存 | 省计算 | 典型方法 |
|:--------|:--------:|:--------:|:--------:|:-----:|:-----:|:--------|
| **W8A16** | INT8(1/2) | FP16 | FP16 | 中 | 低 | MinMax |
| **W6A16** | FP6(3/8) | FP16 | FP16 | 中 | 低 | FP6-LLM |
| **W4A16** | INT4(1/4) | FP16 | FP16 | **高** | 低 | AWQ, GPTQ |
| **W8A8** | INT8(1/2) | INT8(1/2) | INT8 | 中 | **高** | SmoothQuant |
| **W4A8** | INT4(1/4) | INT8(1/2) | INT8 | **高** | **高** | QoQ |
| **W4A4** | INT4(1/4) | INT4(1/4) | INT4 | **高** | **高** | Atom, QuaRot |
| **KV8** | — | — | INT8 | 省KV内存 | — | TensorRT-LLM |
| **KV4** | — | — | INT4 | **省更多** | — | Atom |
| **KV2** | — | — | INT2 | **最省** | — | KIVI, KVQuant |

---

###### 关键递进逻辑

答主原文的分类逻辑其实是：

1. **仅权重量化**（W4A16, W6A16...）
   - 解决"模型太大放不下"的问题 → **省显存**
   - 代价：计算时需反量化回 FP16，计算速度不提升甚至略降

2. **权重激活量化**（W8A8, W4A8, W4A4...）
   - 解决"推理速度不够快"的问题 → **省计算**
   - 代价：激活量化难度大，精度损失风险高

3. **KV Cache 量化**（KV8, KV4, KV2...）
   - 解决"长序列放不下"的问题 → **省 KV 显存**
   - 代价：需要处理逐 token 到达、Attention Sink 等特殊挑战

---

这个框架清晰了吗？需要的话我们继续往下走 **"量化分类"**（PTQ vs QAT vs QAF），或者直接挑一个具体的量化方法完整推演？

## 1.8 量化分类（QAT / QAF / PTQ）

根据应用量化压缩模型的阶段，可以将模型量化分为：

- **量化感知训练（Quantization Aware Training, QAT）**：在模型训练过程中加入伪量化算子，通过训练时统计输入输出的数据范围可以提升量化后模型的精度，适用于对模型精度要求较高的场景。其量化目标无缝地集成到模型的训练过程中，使 LLM 在训练过程中适应低精度表示。
- **量化感知微调（Quantization-Aware Fine-tuning, QAF）**：在微调过程中对 LLM 进行量化。主要目标是确保经过微调的 LLM 在量化为较低位宽后仍保持性能。通过将量化感知整合到微调中，以在模型压缩和保持性能之间取得平衡。
- **训练后量化（Post Training Quantization, PTQ）**：在 LLM 训练完成后对其参数进行量化，只需要少量校准数据，适用于追求高易用性和缺乏训练资源的场景。主要目标是减少 LLM 的存储和计算复杂性，而无需对 LLM 架构进行修改或进行重新训练。PTQ 的主要优势在于其简单性和高效性，但可能会在量化过程中引入一定程度的精度损失。

### 三者怎么区分（核心问题：什么时候做量化？）

| 分类 | 什么时候量化？ | 需不需要训练？ | 精度 |
|:----|:--------------|:-------------|:----|
| **PTQ**（训练后量化） | 模型训练完成后 | 不需要，只需少量校准数据 | 一般 |
| **QAT**（量化感知训练） | 训练过程中 | 需要完整训练 | 最高 |
| **QAF**（量化感知微调） | 微调过程中 | 需要微调数据 | 较高 |

> 🔍 **三阶段类比**
>
> - **PTQ** = 你写好了文章，然后找人翻译成英文。翻译可能不准确，但你没法改原文了。
> - **QAT** = 你写文章时旁边站着一个翻译，你每写一句他就告诉你"这句翻成英文会丢这个信息"，于是你调整中文写法，让翻译后的英文更准确。
> - **QAF** = 文章已写好，但你请了一个校对在旁边盯着，只让你微调几个关键句子来减少翻译损失。

### ① PTQ（训练后量化）—— 直接用数据推演

```
第 1 步：模型已训练完毕（W 已固定，全 FP16）
第 2 步：直接量化（假设 W4A16，per-group）：
  out0 分成两组 [0.5, 0.1 | 0.3, 6.0]
  组A: r_max=0.5, r_min=0.1, s_A=(0.5-0.1)/7=0.0571
    0.5→round(0.5/0.0571)=9→clip到7→7→反量化 0.4   ❌ 误差 0.1
    0.1→round(0.1/0.0571)=2→反量化 0.114            ❌ 误差 0.014
  组B: r_max=6.0, r_min=0.3, s_B=(6.0-0.3)/7=0.814
    0.3→round(0.3/0.814)=0→反量化 0                  ❌ 误差 0.3
    6.0→round(6.0/0.814)=7→反量化 6.0               ✅
第 3 步：量化完了直接部署。整个过程没动 W 的原始值，只是压缩存储。
```

如果用了校准集（如 GPTQ），会拿少量校准数据做补偿调整，但**不重新训练模型主体参数**。

**哪些方法是 PTQ？**

| 方法 | 类型 | 需要校准集吗？ |
|:----|:----|:-------------|
| **RTN** | PTQ | 不需要 |
| **LLM.int8()** | PTQ | 不需要 |
| **GPTQ** | PTQ | 需要 |
| **SmoothQuant** | PTQ | 需要 |
| **AWQ** | PTQ | 需要 |
| **ZeroQuant** | PTQ | 不需要（动态量化） |
| **KIVI** | PTQ | 不需要 |
| **KVQuant** | PTQ | 需要 |

### ② QAT（量化感知训练）—— 完整流程推演

**核心思想**：在训练过程中就"假装"模型已经被量化了，让模型学会适应低精度带来的误差。

**具体做法**：在前向传播中插入**伪量化算子（fake quantization node）**。

```
正常训练的前向：W_fp16 → Y = X · W_fp16ᵀ → Loss → 反向传播
QAT 的前向：    W_fp16 → 伪量化 → W_fake_int4 → 反量化 → W_recon_fp16
                → Y = X · W_recon_fp16ᵀ → Loss → 反向传播
```

**关键**：伪量化是"模拟"量化，不是真的存成 INT4。它把 W_fp16 先压到 INT4 再解压回来，让 W 提前感受到量化误差。

```
训练刚开始时：
W_original = [0.5, 0.1, 0.3, 6.0]
伪量化（INT4 per-group, group=2）：
  组A: s=0.0571 → 0.5→7→0.4, 0.1→2→0.114
  组B: s=0.814 → 0.3→0→0, 6.0→7→6.0
W_recon = [0.4, 0.114, 0, 6.0]  ← 和原始 W 有误差
Y = X · W_reconᵀ  ← 带着误差前向传播
Loss = loss(Y, Y_true)  ← Loss 包含量化误差

反向传播：梯度穿透伪量化节点（STE，Straight-Through Estimator），更新原始 FP16 的 W：
W_new = W_old - lr × grad

经过多轮训练后，W 会逐渐演化成"即使被量化再解压，输出误差也最小"的状态。
比如原始 W 的 6.0 可能变成 5.8，因为 5.8/0.814=7.12→7→5.8，反而更精确了。
```

**QAT 的优缺点**：精度最高；但需要全量训练数据 + 完整训练流程，成本极高。LLM 领域很少用完整的 QAT。

**QAT 训练完成后**：得到被微调过的全精度 W，再做真正的 INT8/INT4 量化部署。因为 W 已在训练中适应量化误差，量化后精度比没有 QAT 的 W 好得多。

### ③ QAF（量化感知微调）—— QLoRA 为代表

**核心思想**：不重新训练整个模型，在微调阶段引入量化感知，用 LoRA 等轻量方法只调一小部分参数。

```
第 1 步：把预训练模型 W 量化到 4bit（NF4 格式），冻结
第 2 步：添加 LoRA 旁路（LoRA_A + LoRA_B，FP16，可学习）
        原始 W 参数量：4×4=16；LoRA 参数量（rank=2）：4×2+2×4=16
        实际大模型中 rank 很小（如 8），LoRA 只占 0.1%~1%
第 3 步：前向：Y = X · W_fp16_reconᵀ + X · (LoRA_A · LoRA_B)ᵀ
        ↑量化部分冻结            ↑LoRA 部分可学习
第 4 步：反向：梯度只更新 LoRA_A 和 LoRA_B，W_int4 不动
```

**为什么 QLoRA 用 NF4 而不是 INT4？**

```
W 所有值：[0.5,0.1,0.3,6.0,0.4,0.2,0.6,0.8,0.7,0.3,0.5,0.2,0.1,0.9,0.4,0.6]
去掉离群值 6.0，大部分值集中在 [0.1, 0.9]

INT4 均匀量化：s=(6.0-0.1)/15=0.393 → 0.1~0.9 只分到 0.8/0.393≈2 个格子
NF4 非线性量化：中间密两边疏 → 0.1~0.9 分到更多格子，小值精度好
```

**三者在 LLM 领域的实际地位**（用 out0 行 [0.5, 0.1, 0.3, 6.0] 对比）：

```
PTQ（GPTQ）：   量化后 [0.4, 0.0, 0.0, 6.0]  ← 直接量化，快，精度一般
QAT（LLM-QAT）：训练后 W=[0.48,0.12,0.28,5.9] ← W 被微调了，量化后误差更小，成本最高
QAF（QLoRA）：  量化后冻结 W_int4，加 LoRA 旁路补偿 ← 成本低，精度中上
```

> 🔍 **学习追问：GPTQ / AWQ / SmoothQuant 都是 PTQ 吗？**
>
> **对，三者都是 PTQ。** 但它们"动没动权重值"不一样：
>
> ```
> PTQ 家族
> ├── 不动原始 W 值（纯数学变换 + 直接量化）
> │   ├── SmoothQuant — 平滑因子，把激活难度转给权重
> │   └── AWQ — 根据激活分布，找最佳缩放因子保护显著通道
> └── 动了原始 W 值（用校准集做优化补偿）
>     └── GPTQ — 量化一个，调整其他来补误差
> ```
>
> 三者都不需要重新训练模型，都不需要反向传播，都只需要少量校准数据（甚至不需要，如 LLM.int8()），所以都归为 PTQ。

---

### 📌 原始 Q&A：量化分类 QAT/QAF/PTQ（历史第 30-32 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 30 轮）

量化分类
这部分的内容

##### A（历史第 31 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！继续用我们的 4×4 数据来推演。

---

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

激活 X(2×4)：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]
```

---

###### 核心问题：什么时候做量化？

答主原文把量化分为三类，区分标准是 **"量化的阶段"**：

| 分类 | 什么时候量化？ | 需不需要训练？ | 精度 |
|:----|:--------------|:-------------|:----|
| **PTQ**（训练后量化） | 模型训练完成后 | 不需要，只需少量校准数据 | 一般 |
| **QAT**（量化感知训练） | 训练过程中 | 需要完整训练 | 最高 |
| **QAF**（量化感知微调） | 微调过程中 | 需要微调数据 | 较高 |

---

###### 一、PTQ（Post Training Quantization）— 训练后量化

###### 核心思想

> 模型已经训好了，我直接拿过来量化。最多拿一点校准数据调一下 scale，不去动模型参数。

###### 用我们数据推演

**第 1 步：模型已训练完毕**

```
W 已经固定了，全 FP16 精度：
out0: [0.5, 0.1, 0.3, 6.0]  ← FP16
out1: [0.4, 0.2, 0.6, 0.8]  ← FP16
out2: [0.7, 0.3, 0.5, 0.2]  ← FP16
out3: [0.1, 0.9, 0.4, 0.6]  ← FP16
```

**第 2 步：直接量化**

假设我们做 W4A16（仅权重量化），用 per-group 方式：

```
out0 分成两组： [0.5, 0.1 | 0.3, 6.0]

组A: r_max=0.5, r_min=0.1, s_A=(0.5-0.1)/7=0.0571
  0.5→round(0.5/0.0571)=9→clip到7→7→反量化 0.4   ❌ 误差 0.1
  0.1→round(0.1/0.0571)=2→反量化 0.114            ❌ 误差 0.014

组B: r_max=6.0, r_min=0.3, s_B=(6.0-0.3)/7=0.814
  0.3→round(0.3/0.814)=0→反量化 0                  ❌ 误差 0.3
  6.0→round(6.0/0.814)=7→反量化 6.0               ✅
```

**量化完了，直接部署。** 整个过程没动过 W 的原始值，只是把 W 压缩了存。

###### 如果用了校准集呢？

有些 PTQ 方法（如 GPTQ）会拿少量校准数据来微调量化后的权重：

```
校准集 X_calib = [1.0, 0.5, 0.3, 0.2]（假设就一条样本）

GPTQ 的做法：
  量化某个权重后，调整同组内其他未量化的权重，使得输出误差最小。
  量化 0.3 后 → 调整 6.0 变成 5.8 → 使 Y 的误差变小
```

**但注意**：PTQ 的校准只是调整 **量化参数** 或 **做少量补偿**，不会去重新训练模型的主体参数。

###### PTQ 的优缺点

**优点**：
- 不需要训练，快
- 只需要少量校准数据（甚至不需要，如 LLM.int8()）
- 答主原文说：**"追求高易用性和缺乏训练资源的场景"**

**缺点**：
- 精度损失相对较大
- 答主原文说：**"可能会在量化过程中引入一定程度的精度损失"**

###### 哪些方法是 PTQ？

| 方法 | 类型 | 需要校准集吗？ |
|:----|:----|:-------------|
| **RTN** | PTQ | 不需要 |
| **LLM.int8()** | PTQ | 不需要 |
| **GPTQ** | PTQ | 需要 |
| **SmoothQuant** | PTQ | 需要 |
| **AWQ** | PTQ | 需要 |
| **ZeroQuant** | PTQ | 不需要（动态量化） |
| **KIVI** | PTQ | 不需要 |
| **KVQuant** | PTQ | 需要 |

---

###### 二、QAT（Quantization Aware Training）— 量化感知训练

###### 核心思想

> 在训练过程中就"假装"模型已经被量化了，让模型学会适应低精度带来的误差。

###### 具体怎么做

在训练的前向传播中，**插入伪量化算子（fake quantization node）**：

```
正常训练的前向：
  W_fp16 → 计算 Y = X · W_fp16ᵀ → 算 Loss → 反向传播

QAT 的前向：
  W_fp16 → 伪量化 → W_fake_int4 → 反量化 → W_recon_fp16
  → 计算 Y = X · W_recon_fp16ᵀ → 算 Loss → 反向传播
```

**关键**：伪量化是"模拟"量化，不是真的存成 INT4。它只是把 W_fp16 先压到 INT4 再解压回来，让 W 提前感受到量化误差。

###### 用我们数据推演

**训练刚开始时**：

```
W_original = [0.5, 0.1, 0.3, 6.0]

伪量化（INT4 per-group, group=2）：
[0.5, 0.1 | 0.3, 6.0]
组A: s=0.0571 → 0.5→7→0.4, 0.1→2→0.114
组B: s=0.814 → 0.3→0→0, 6.0→7→6.0

W_recon = [0.4, 0.114, 0, 6.0]  ← 和原始 W 有误差

Y = X · W_reconᵀ  ← 带着误差前向传播
Loss = loss(Y, Y_true)  ← 算出来的 Loss 包含了量化误差
```

**反向传播时**：

梯度正常回传到 W，但 **伪量化节点本身不传递梯度**（它是不可微的），梯度直接"穿透"它：

```
W_new = W_old - lr × grad  ← 更新的是原始 FP16 的 W

W_new = [0.51, 0.11, 0.31, 5.99]  ← 微调了原始权重
```

**经过多轮训练后**：

```
W 逐渐适应了"自己会被量化"这件事：
原始 W 会朝"即使被量化成 INT4，误差也最小"的方向演化

比如：
原始 W 的 6.0 可能会变成 5.8，因为 5.8/0.814=7.12→7→5.8，反而更精确了
```

###### 用类比理解

- **PTQ** = 你写好了文章，然后找人翻译成英文。翻译可能不准确，但你没法改原文了。
- **QAT** = 你写文章的时候，旁边就站着一个翻译，你每写一句他就告诉你"这句翻成英文会丢这个信息"，于是你调整中文写法，让翻译后的英文更准确。

###### QAT 的优缺点

**优点**：
- 精度最高
- 答主原文说：**"适用于对模型精度要求较高的场景"**

**缺点**：
- 需要完整训练，成本极高
- 答主原文说：**"在模型训练过程中加入伪量化算子"**

**对于 LLM 来说**：QAT 需要 **全量训练数据 + 完整训练流程**，成本太高。所以 LLM 领域很少用完整的 QAT。

---

###### 三、QAF（Quantization-Aware Fine-tuning）— 量化感知微调

###### 核心思想

> 不重新训练整个模型，而是在微调阶段引入量化感知。用 LoRA 等轻量方法，只调一小部分参数。

###### 和 QAT 的区别

| | QAT | QAF |
|:--|:----|:----|
| 阶段 | 从头训练 | 微调（在预训练模型基础上） |
| 数据量 | 全量训练数据 | 少量微调数据 |
| 训练成本 | 极高 | 较低 |
| 典型方法 | 传统 QAT | **QLoRA** |

###### QLoRA 是怎么做的

答主原文对 QLoRA 的描述：

> QLoRA 将预训练模型量化为 4 比特，然后添加一小组可学习的 LoRA 权重，这些权重通过量化权重的反向传播梯度进行微调。

**第 1 步：把预训练模型 W 量化到 4bit**

```
W_original(full precision) → W_int4(NF4格式)

out0: [0.5, 0.1, 0.3, 6.0] → NF4 量化
```

**第 2 步：冻结 W_int4，添加 LoRA 旁路**

```
冻结：W_int4(4×4) ← 不动了，就存成 INT4

新增：LoRA_A(4×2) + LoRA_B(2×4) ← 可学习，FP16

LoRA_A 和 LoRA_B 很小：
  原始 W 参数量：4×4 = 16
  LoRA 参数量：4×2 + 2×4 = 16
  如果 rank=2，LoRA 的参数量和原始 W 一样
  但实际中 rank 很小（如 8），对大模型来说 LoRA 只占 0.1%~1%
```

**第 3 步：微调时，前向传播**

```
计算 Y 时：
  W_int4 → 反量化 → W_fp16_recon
  Y = X · W_fp16_reconᵀ + X · (LoRA_A · LoRA_B)ᵀ
      ↑ 量化部分冻结        ↑ LoRA 部分可学习
```

**第 4 步：反向传播时**

```
梯度只更新 LoRA_A 和 LoRA_B
W_int4 不动
```

###### 为什么 QLoRA 用 NF4 而不是 INT4？

答主原文说：

> NF4 对正态分布数据比 INT4/FP4 更优

用我们 W 的数据看：

```
W 所有值分布：[0.5, 0.1, 0.3, 6.0, 0.4, 0.2, 0.6, 0.8, 0.7, 0.3, 0.5, 0.2, 0.1, 0.9, 0.4, 0.6]

去掉离群值 6.0，大部分值集中在 [0.1, 0.9]：
  0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 6.0

INT4 均匀量化：s = (6.0-0.1)/15 = 0.393
  → 0.1~0.9 只分到 0.8/0.393 ≈ 2 个格子

NF4 非线性量化：中间密，两边疏
  → 0.1~0.9 分到更多格子，小值精度好
  → 6.0 虽然分到的格子少，但它本身是离群值，丢了也不太影响
```

###### QAF 的优缺点

**优点**：
- 训练成本低（只微调 LoRA）
- 精度比 PTQ 好
- 答主原文说：**"在模型压缩和保持性能之间取得平衡"**

**缺点**：
- 还是需要微调数据和计算资源
- 比纯 PTQ 复杂

---

###### 三者在 LLM 领域的实际地位

用我们 4×4 的 W，看三种方法分别怎么处理 out0 行的 [0.5, 0.1, 0.3, 6.0]：

```
PTQ（GPTQ）：
  量化前：    [0.5, 0.1, 0.3, 6.0]
  量化后：    [0.4, 0.0, 0.0, 6.0]  ← 直接量化，小值误差大，但快
  只需要校准数据，不改权重

QAT（LLM-QAT）：
  训练开始前：W=[0.5, 0.1, 0.3, 6.0]
  训练过程中：每次前向都做伪量化，让 W 感知误差
  训练结束后：W=[0.48, 0.12, 0.28, 5.9]  ← W 被微调了，量化后误差更小
  需要全量训练数据，成本高

QAF（QLoRA）：
  量化后冻结：W_int4=[0.4, 0.0, 0.0, 6.0]（同样量化误差）
  但新增：LoRA 旁路，微调 LoRA 来补偿量化误差
  最终输出：Y = X·W_int4_reconᵀ + X·LoRAᵀ
           ↑ 有误差     ↑ 补偿误差
  只需要微调数据，成本低
```

###### 一句话总结

| 方法 | 一句话 | 对 W 的改动 | 成本 |
|:----|:------|:-----------|:----|
| **PTQ** | 训完直接量化，最多调调 scale | 不动原始 W | 最低 |
| **QAT** | 训练时就假装被量化了，提前适应 | 训练过程中微调 W | 最高 |
| **QAF** | 量化后加个 LoRA 旁路来补偿误差 | 不动 W，新增 LoRA 参数 | 中 |

---

###### 答主原文的原文

> **量化感知训练（QAT）**：在模型训练过程中加入伪量化算子，通过训练时统计输入输出的数据范围可以提升量化后模型的精度，适用于对模型精度要求较高的场景。

> **量化感知微调（QAF）**：在微调过程中对LLM进行量化。主要目标是确保经过微调的LLM在量化为较低位宽后仍保持性能。

> **训练后量化（PTQ）**：在LLM训练完成后对其参数进行量化，只需要少量校准数据，适用于追求高易用性和缺乏训练资源的场景。

---

这个框架清晰了吗？如果清楚了，我们接下来可以继续看 **"模型量化方法精要"** 部分，逐个看每个方法的具体原理和计算过程。

##### Q（历史第 32 轮）


---
所以GPTQ和AWQ和SmoothQuant都是PTQ？

##### A（历史第 33 轮，正文原样保留；仅将内部标题降级以保持目录层级）

**对，三者都是 PTQ（训练后量化）。**

不过它们的 PTQ 方式有细微差别，用我们 4×4 的数据来看：

---

###### 三者都是 PTQ 的证据

| 方法 | 核心特征 | 是不是 PTQ？ |
|:----|:---------|:-----------|
| **SmoothQuant** | 用校准集算出平滑因子 s，乘到 W 上，X 除个 s，然后直接 RTN 量化 | ✅ PTQ，不动模型参数 |
| **AWQ** | 用校准集算出每个通道的最佳缩放因子 α，保护显著权重，然后直接量化 | ✅ PTQ，不动模型参数 |
| **GPTQ** | 用校准集算出 Hessian 矩阵，量化一个权重后调整其他权重来补偿误差 | ✅ PTQ，**但动了权重值** |

---

###### 同是 PTQ，但"动没动权重值"不一样

###### SmoothQuant — 完全不动 W 的原始值

```
W_out0: [0.5, 0.1, 0.3, 6.0]

SmoothQuant 做的事：
  1. 从校准集算出一个平滑因子 s（比如 s=0.5）
  2. W' = W × s = [0.25, 0.05, 0.15, 3.0]
  3. X' = X / s
  4. 然后 W' 做 RTN 量化

注意：W' 不是 W 的原始值了，但 SmoothQuant 是在量化前做的"数学等价变换"，
并不是"微调"权重。Y = X'·W'ᵀ = (X/s)·(W×s)ᵀ = X·Wᵀ，数学上等价。
```

###### AWQ — 完全不动 W 的原始值

```
W_out0: [0.5, 0.1, 0.3, 6.0]

AWQ 做的事：
  1. 从校准集看激活分布，发现某些通道是显著通道
  2. 用 grid search 找到最佳 α
  3. W' = W × diag(s_X^α)，然后做 RTN 量化
  4. X' = X × diag(s_X^(-α))

和 SmoothQuant 一样，W' 也不是原始值了，但也是"数学等价变换"。
AWQ 没有通过梯度下降去改 W。
```

###### GPTQ — 真正改了 W 的原始值

```
W_out0: [0.5, 0.1, 0.3, 6.0]

GPTQ 做的事：
  1. 拿校准集过一遍，算 Hessian 矩阵
  2. 量化某个权重后，调整同组内其他未量化的权重来补偿误差

比如：
  量化 0.5 → INT4 → 反量化后变成 0.4（误差 -0.1）
  于是调整同组的 0.1 变成 0.15，来补偿这个 -0.1 的误差

最终 W 被改了：
  原始 W：  [0.5, 0.1, 0.3, 6.0]
  量化后 W：[0.4, 0.15, 0.3, 6.0]  ← 0.15 是被调整过的
```

---

###### 所以它们三个的区别可以这样看

```
PTQ 家族
├── 不动原始 W 值（纯数学变换 + 直接量化）
│   ├── SmoothQuant — 平滑因子，把激活难度转给权重
│   └── AWQ — 根据激活分布，找最佳缩放因子保护显著通道
│
└── 动了原始 W 值（用校准集做优化补偿）
    └── GPTQ — 量化一个，调整其他来补误差
```

但 **三者都不需要重新训练模型**，都不需要反向传播，都只需要少量校准数据（甚至不需要，如 LLM.int8()），所以都归为 **PTQ**。

---

###### 对比 QAT 的话就更清楚了

```
QAT — 需要完整训练，需要反向传播，需要全量数据
  └─ 比如 LLM-QAT：训练过程中插入伪量化节点，让 W 学会适应量化误差
  └─ 成本：极高

PTQ — 不需要训练，不需要反向传播，只需要少量校准数据
  ├─ SmoothQuant： 几十分钟搞定
  ├─ AWQ：        几十分钟搞定
  └─ GPTQ：       几十分钟到几小时搞定
  └─ 成本：极低

QAF — 介于两者之间，量化后加 LoRA 微调
  └─ 比如 QLoRA：量化后加 LoRA 旁路，只调 LoRA 参数
  └─ 成本：中等
```

所以结论很明确：**GPTQ、AWQ、SmoothQuant 都是 PTQ**，只是 PTQ 内部的策略不同。

# 第二部分：模型量化方法精要

> **本部分结构**：先给出博主原文的方法总览（含 18 个方法速览表），再按方法逐个展开——每个方法都包含【博主原文要点】+【完整数值推演】+【学习追问】+【易错校正】。数值推演部分保留历史学习过程记录（Q&A 推导顺序），是学习过程的补充，不等同于博主原文。

最近两年，业界诞生了很多关于 LLM 量化方法。LLM.int8() 和 SmoothQuant 都属于 round-to-nearest (RTN) 量化。针对激活中的离群值，SmoothQuant 给出了与 LLM.int8() 不同的解题思路——既然激活的量化比权重的量化难得多，那么可以通过一个平滑系数，把二者的难度综合一下。GPTQ 则是把量化问题视作优化问题，逐层寻找最优的量化权重。AWQ 方法则基于不是所有权重都是同等重要的这一发现，其中只有 0.1%-1% 的权重（salient weights）对模型精度贡献更大，并且这些权重取决于激活值分布而不是权重分布。

> **方法速览表（共 18 个方法，ZeroQuant 系列合并为 2.9 一节）**

| 方法 | 类型 | 核心思想 |
|------|------|----------|
| **RTN** | 仅权重 | 四舍五入近似到最近整数 |
| **LLM-QAT** | QAT+无数据蒸馏 | 保留异常值，对称MinMax量化，逐通道/逐Token |
| **QLoRA** | 量化感知微调 | NF4+双量化+分页优化器，4bit微调 |
| **LLM.int8()** | W8A8混合精度 | 异常值用FP16，其余INT8 |
| **GPTQ** | W4A16 | 把量化当优化问题，Cholesky分解+Hessian矩阵 |
| **SmoothQuant** | W8A8 | 平滑因子将量化难度从激活转移到权重 |
| **AWQ** | W4A16 | 保护Top显著权重，grid search找最佳缩放因子α |
| **SpQR** | 仅权重 | 隔离异常值+双层量化（量化元数据再量化） |
| **ZeroQuant** | W8A8 | 分组权重+逐Token动态激活，无需校准集 |
| **ZeroQuant-V2** | W4A8 | +LoRC低秩补偿（新增两个低秩矩阵近似量化误差） |
| **ZeroQuant-FP** | W4A8 | FP4/FP8+位移（bit-shifting）替代反量化 |
| **FP6-LLM** | W6A16 | TC-FPx kernel，Tensor Core+SIMT混合计算 |
| **KIVI** | KV2 | KV缓存分组量化，Key逐通道/Value逐Token |
| **IntactKV** | KV | 保持关键词元（公共前缀）的KV无损 |
| **KVQuant** | KV2-KV3 | 逐通道量化Key+RoPE前量化+Per-Vector稀疏量化 |
| **Atom** | W4A4KV4 | 混合精度+通道重排序+异常值INT8 |
| **QuaRot** | W4A4KV4 | Hadamard旋转消除异常值 |
| **QoQ** | W4A8KV4 | 渐进式分组量化+SmoothAttention+旋转矩阵 |

---

**博主原文配图**（源自知乎原文）：

![原文图 3](https://picx.zhimg.com/v2-621c9e3cbcf269a89d5afeb96f561927_r.jpg?source=2c26e567)


---

### 📌 原始 Q&A：模型量化方法精要总览（历史第 34 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 34 轮）

模型量化方法精要
 这部分的内容呢

##### A（历史第 35 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！这部分是答主回答的核心，我们来逐个方法用 4×4 数据推演。

---

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

激活 X(2×4)：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]
```

---

###### ① RTN（Round to Nearest）

###### 原理

> 四舍五入，把量化权重近似到最近的整数上。

###### 用我们数据推演

对 out0 行 [0.5, 0.1, 0.3, 6.0] 做 INT4 量化：

```
INT4 对称量化（范围 -8 ~ 7）：

s = max(|0.5|, |0.1|, |0.3|, |6.0|) / 7 = 6.0 / 7 ≈ 0.857

0.5 → round(0.5/0.857) = round(0.58) = 1  → 反量化 0.857  ❌ 误差 0.357
0.1 → round(0.1/0.857) = round(0.12) = 0  → 反量化 0       ❌ 误差 0.1
0.3 → round(0.3/0.857) = round(0.35) = 0  → 反量化 0       ❌ 误差 0.3
6.0 → round(6.0/0.857) = round(7.0) = 7   → 反量化 6.0     ✅ 精确

Y[0][0] = X[token0]·W_recon[out0]
        = 2.0×0.857 + 0.5×0 + 0.8×0 + 0.3×6.0
        = 1.714 + 0 + 0 + 1.8 = 3.514

原始 Y[0][0] = 3.09
误差 = 3.514 - 3.09 = 0.424  ❌
```

###### 答主怎么说

> 对于LLM而言，由于大量异常值的存在，如果单独使用RTN量化，对精度影响较大。通常会采用一些方式抑制异常值之后，再使用RTN量化。

SmoothQuant、AWQ 本质上都是"先抑制异常值，再做 RTN"。

---

###### ② LLM-QAT

###### 原理

> 在训练过程中加入伪量化，让模型提前适应低精度。但用的是 **知识蒸馏**（无数据），而不是原始训练数据。

###### 用我们数据看它怎么做

**选择对称 MinMax 量化**，因为答主说：

> 在具有门控线性单元（GLU）的模型中，激活权重大多是对称分布的。

**量化策略**：
- 权重：per-channel 量化
- 激活：per-token 量化
- KV 缓存：per-token 量化

```
权重 per-channel 量化：
out0: [0.5, 0.1, 0.3, 6.0] → s₀ = 6.0/127 = 0.0472
out1: [0.4, 0.2, 0.6, 0.8] → s₁ = 0.8/127 = 0.0063
out2: [0.7, 0.3, 0.5, 0.2] → s₂ = 0.7/127 = 0.0055
out3: [0.1, 0.9, 0.4, 0.6] → s₃ = 0.9/127 = 0.0071

激活 per-token 量化：
token0: [2.0, 0.5, 0.8, 0.3] → s_token0 = 2.0/127 = 0.0157
token1: [0.4, 7.0, 0.2, 0.6] → s_token1 = 7.0/127 = 0.0551
```

**关键**：LLM-QAT 是在训练中做的，所以 W 会在训练过程中被微调，让"量化后的误差"最小化。这和 PTQ 的"一次性量化"不同。

###### 答主的关键句

> 在训练的初始阶段，任何基于裁剪的方法都会导致异常高的困惑度...因此，选择保留这些异常值。

---

###### ③ QLoRA

###### 原理

> 预训练模型量化为 4bit，然后加一小队 LoRA 权重来微调补偿。

###### 用我们数据推演

**第 1 步：把 W 量化到 NF4（4bit）**

NF4 不是 INT4，它是专门为正态分布设计的非线性量化：

```
NF4 的 16 个格子：
[-1.0, -0.696, -0.525, -0.393, -0.277, -0.174, -0.085, -0.008,
  0.008, 0.085, 0.174, 0.277, 0.393, 0.525, 0.696, 1.0]

先把 W 归一化到 [-1, 1]：
out0: [0.5, 0.1, 0.3, 6.0]
  归一化因子 = max(|0.5|,|0.1|,|0.3|,|6.0|) = 6.0
  归一化后 = [0.083, 0.017, 0.05, 1.0]

  0.083 → 离 0.085 最近 → 索引 9  → 反量化 0.085
  0.017 → 离 0.008 最近 → 索引 8  → 反量化 0.008
  0.05  → 离 0.085 最近 → 索引 9  → 反量化 0.085
  1.0   → 离 1.0 最近  → 索引 15 → 反量化 1.0

  反量化还原：× 6.0
  [0.51, 0.048, 0.51, 6.0]  ← 比 INT4 的 [0.857, 0, 0, 6.0] 好多了
```

**第 2 步：冻结 W_int4，加 LoRA**

```
W_int4 冻结不动

新增 LoRA_A(4×2) 和 LoRA_B(2×4)：
  LoRA_A 随机初始化，LoRA_B 初始化为 0

前向传播：
  Y = X · W_int4_reconᵀ + X · (LoRA_A · LoRA_B)ᵀ
      ↑ 量化部分冻结        ↑ LoRA 补偿

反向传播只更新 LoRA_A 和 LoRA_B
```

###### 答主提到的两个关键技术

**双量化（Double Quantization）**：
> 对第一次量化后的那些常量（scale）再进行一次量化

```
W 量化后，每组的 scale 也要存。
比如 128 个一组，每组一个 FP32 scale → 额外 32bit/128个参数 = 0.25bit/参数
双量化就是把 scale 也量化了，比如从 FP32 量化到 FP8 → 额外开销降到 0.0625bit/参数
```

**分页优化器（Paged Optimizers）**：
> 防止梯度检查点期间的内存峰值

优化器状态放不下 GPU 时，自动卸载到 CPU 内存，需要时再加载回来。

---

###### ④ LLM.int8()

###### 原理

> 混合精度分解：大部分用 INT8，离群值用 FP16。

###### 用我们数据推演

**第 1 步：检测离群值列**

```
X = [2.0, 0.5, 0.8, 0.3]
    [0.4, 7.0, 0.2, 0.6]

按列看，c1 列有 7.0，超过阈值（比如 6.0）→ 标记为离群列
c3 列 6.0 也超过阈值 → 也标记为离群列

离群列：c1, c3
正常列：c0, c2
```

**第 2 步：拆分矩阵**

```
正常部分（INT8 计算）：
  X_norm = [2.0, 0.8]  ← 只保留 c0, c2 列
           [0.4, 0.2]
  
  W_norm = [0.5, 0.3]  ← 只保留 c0, c2 列
           [0.4, 0.6]
           [0.7, 0.5]
           [0.1, 0.4]

  量化到 INT8 → 做 INT8 矩阵乘

离群部分（FP16 计算）：
  X_outlier = [0.5, 0.3]  ← 只保留 c1, c3 列
              [7.0, 0.6]
  
  W_outlier = [0.1, 6.0]
              [0.2, 0.8]
              [0.3, 0.2]
              [0.9, 0.6]

  保持 FP16 → 做 FP16 矩阵乘
```

**第 3 步：合并结果**

```
Y = INT8_result + FP16_result
```

###### 答主怎么说

> LLM.int8() 的推理性能差

因为每一层都要做：检测离群值 → 拆分矩阵 → INT8 算 → FP16 算 → 合并。开销大。

---

###### ⑤ GPTQ

###### 原理

> 把量化看作优化问题：量化一个权重后，调整同组内其他未量化的权重来补偿误差。

###### 用我们数据推演

**第 1 步：准备校准数据，算 Hessian 矩阵**

```
假设校准集就一条样本 X_calib = [1.0, 0.5, 0.3, 0.2]

对于 out0 行 [0.5, 0.1, 0.3, 6.0]：
这一行的 Hessian 矩阵 H = 2·X_calibᵀ·X_calib

H = 2 × [1.0]   × [1.0, 0.5, 0.3, 0.2]
         [0.5]
         [0.3]
         [0.2]

H = 2 × [1.0, 0.5, 0.3, 0.2]
         [0.5, 0.25, 0.15, 0.1]
         [0.3, 0.15, 0.09, 0.06]
         [0.2, 0.1, 0.06, 0.04]
```

**第 2 步：分组量化（group=2）**

```
out0: [0.5, 0.1 | 0.3, 6.0]

先量化组A的 0.5：
  INT4 量化：0.5 → 1 → 反量化 0.857（误差 -0.357）
  然后调整组A内未量化的 0.1：
  调整量 = Hessian 信息 × 量化误差
  0.1 → 0.1 + 0.5 × (-0.357) ≈ 0.1 - 0.179 ≈ -0.079

再量化组A的 -0.079（调整后的值）：
  INT4 量化：-0.079 → 0 → 反量化 0（误差 0.079）
  组A没有更多未量化元素了，不调整了

组A量化完成：[0.5, 0.1] → [0.857, 0]

再量化组B的 0.3：
  INT4 量化：0.3 → 0 → 反量化 0（误差 0.3）
  调整组B内未量化的 6.0：
  6.0 → 6.0 + 0.5 × 0.3 = 6.15

再量化组B的 6.15：
  INT4 量化：6.15 → 7 → 反量化 6.0（误差 0.15）

out0 量化结果：[0.857, 0, 0, 6.0]
```

**注意**：这是简化演示。实际 GPTQ 用的是 Cholesky 分解求 Hessian 逆矩阵，然后一次处理多个列（Lazy Batch-Updates），不是逐元素调整。

###### 答主说的 GPTQ 创新点

**① 取消贪心算法**：
OBS（之前的做法）要挑"对目标影响最小的参数"先量化，GPTQ 发现直接按顺序量化也行，这样每行可以做并行计算。

**② Lazy Batch-Updates**：
一次处理 128 列，而不是逐列更新，减少 GPU 内存带宽压力。

**③ Cholesky 分解**：
提前算好 Hessian 矩阵的逆，后续更新时直接查，减少计算量。

---

###### ⑥ SmoothQuant

###### 原理

> 激活难量化（有离群值），权重容易量化。通过一个平滑因子 s，把激活的量化难度"转移"到权重上。

###### 用我们数据推演

**第 1 步：算平滑因子 s**

```
s_j = 1 - α × (max|X_j| / max|X|) + α

其中 α 是平滑强度（默认 0.5），X_j 是第 j 列激活值

X 的每列最大值：
  c0: max(|2.0|,|0.4|) = 2.0
  c1: max(|0.5|,|7.0|) = 7.0
  c2: max(|0.8|,|0.2|) = 0.8
  c3: max(|0.3|,|0.6|) = 0.6

max|X| = 7.0

s₀ = 1 - 0.5 × (2.0/7.0) + 0.5 = 1 - 0.143 + 0.5 = 1.357
s₁ = 1 - 0.5 × (7.0/7.0) + 0.5 = 1 - 0.5 + 0.5 = 1.0
s₂ = 1 - 0.5 × (0.8/7.0) + 0.5 = 1 - 0.057 + 0.5 = 1.443
s₃ = 1 - 0.5 × (0.6/7.0) + 0.5 = 1 - 0.043 + 0.5 = 1.457
```

**第 2 步：应用平滑变换**

```
W' = W × diag(s)   权重乘以 s
X' = X / diag(s)   激活除以 s

W'：
  out0: [0.5×1.357, 0.1×1.0, 0.3×1.443, 6.0×1.457]
       = [0.679, 0.1, 0.433, 8.742]
  out1: [0.4×1.357, 0.2×1.0, 0.6×1.443, 0.8×1.457]
       = [0.543, 0.2, 0.866, 1.166]
  out2: [0.7×1.357, 0.3×1.0, 0.5×1.443, 0.2×1.457]
       = [0.950, 0.3, 0.722, 0.291]
  out3: [0.1×1.357, 0.9×1.0, 0.4×1.443, 0.6×1.457]
       = [0.136, 0.9, 0.577, 0.874]

X'：
  token0: [2.0/1.357, 0.5/1.0, 0.8/1.443, 0.3/1.457]
         = [1.474, 0.5, 0.554, 0.206]
  token1: [0.4/1.357, 7.0/1.0, 0.2/1.443, 0.6/1.457]
         = [0.295, 7.0, 0.139, 0.412]
```

**第 3 步：看发生了什么变化**

```
平滑前 X 的离群值：
  c1 列有 7.0，c0 列有 2.0，范围差异大

平滑后 X' 的离群值：
  c1 列 7.0 没变（s₁=1.0），但 c0 列从 2.0 降到了 1.474
  激活的离群值被"压"了

平滑前 W 的离群值：
  out0 的 c3 列有 6.0

平滑后 W' 的离群值：
  out0 的 c3 列从 6.0 变成了 8.742 ← 权重更难量化了！
  但权重本来就好量化，8.742 也没关系
```

**关键洞察**：平滑后激活的量化难度降低了，但权重的量化难度升高了。**但权重本来就容易量化，所以整体效果是好的。**

###### 数学等价性

```
Y = X · Wᵀ
  = (X / diag(s)) · (W × diag(s))ᵀ
  = X' · W'ᵀ  ✅  数学上完全等价
```

###### 答主说的三种策略

| 策略 | 权重量化 | 激活量化 | 性能 |
|:----|:--------|:--------|:----|
| **O1** | per-channel INT8 | per-token INT8 | 精度最高，速度一般 |
| **O2** | per-channel INT8 | per-tensor INT8 | 精度好，速度快 |
| **O3** | per-tensor INT8 | per-tensor INT8 | 精度下降 0.8%，最快 |

---

###### ⑦ AWQ

###### 原理

> 不是所有权重都同等重要。约 0.1%-1% 的"显著权重"对模型精度贡献更大。这些显著权重由**激活分布**决定，不是权重分布。

###### 用我们数据推演

**第 1 步：找显著通道**

```
根据激活分布找显著通道：

X 的每列平均幅值（绝对值）：
  c0: (|2.0| + |0.4|) / 2 = 1.2
  c1: (|0.5| + |7.0|) / 2 = 3.75  ← 最大！
  c2: (|0.8| + |0.2|) / 2 = 0.5
  c3: (|0.3| + |0.6|) / 2 = 0.45

c1 的激活幅值最大 → c1 对应的权重通道最显著
```

**第 2 步：计算缩放因子 s_X**

```
s_X = 激活每列的平均幅值
s_X = [1.2, 3.75, 0.5, 0.45]
```

**第 3 步：用 grid search 找最佳 α**

```
s = s_X^α, α ∈ [0, 1]

AWQ 做 grid search，比如试 α = 0, 0.1, 0.2, ..., 1.0

α=0 时：s = [1, 1, 1, 1] → 不做缩放，等于 RTN
α=0.5 时：s = [1.2^0.5, 3.75^0.5, 0.5^0.5, 0.45^0.5]
           = [1.095, 1.936, 0.707, 0.671]
α=1.0 时：s = [1.2, 3.75, 0.5, 0.45] → 最激进的缩放

选使量化后输出误差最小的 α
```

**第 4 步：应用缩放**

```
假设选到 α=0.5：

W' = W × diag(s)
   out0: [0.5×1.095, 0.1×1.936, 0.3×0.707, 6.0×0.671]
        = [0.548, 0.194, 0.212, 4.026]
   out1: [0.4×1.095, 0.2×1.936, 0.6×0.707, 0.8×0.671]
        = [0.438, 0.387, 0.424, 0.537]
   out2: [0.7×1.095, 0.3×1.936, 0.5×0.707, 0.2×0.671]
        = [0.767, 0.581, 0.354, 0.134]
   out3: [0.1×1.095, 0.9×1.936, 0.4×0.707, 0.6×0.671]
        = [0.110, 1.742, 0.283, 0.403]

X' = X / diag(s)
   token0: [2.0/1.095, 0.5/1.936, 0.8/0.707, 0.3/0.671]
          = [1.826, 0.258, 1.132, 0.447]
   token1: [0.4/1.095, 7.0/1.936, 0.2/0.707, 0.6/0.671]
          = [0.365, 3.616, 0.283, 0.894]
```

**观察**：
- c1（显著通道）：s>1，W 被放大 → 量化时更精确（因为值越大，相对误差越小）
- c2, c3（非显著通道）：s<1，W 被缩小 → 量化时精度略降，但反正是非显著通道，影响小

###### 和 SmoothQuant 的区别

| | SmoothQuant | AWQ |
|:--|:-----------|:----|
| s 怎么算？ | 公式计算：s = 1 - α×max|X_j|/max|X| + α | 激活幅值的 α 次方：s = s_X^α |
| α 怎么选？ | 固定值（默认 0.5） | grid search 搜索最优 α |
| 目标 | 让激活的离群值变小 | 保护显著通道（激活大的通道） |
| 本质 | 平滑激活的离群值 | 按重要性分配精度 |

---

###### ⑧ SpQR

###### 原理

> 隔离离群权重，用高精度存；其他权重用低精度存。量化元数据（scale）本身也量化。

###### 用我们数据推演

**第 1 步：找离群权重**

```
W 中离群值检测：
out0: [0.5, 0.1, 0.3, 6.0]  ← 6.0 是离群值
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

把 6.0 隔离出来，单独存为 FP16（用稀疏矩阵存）
剩下的非离群值用 3bit 量化
```

**第 2 步：双层量化**

```
分组量化（group=16），然后对 scale 再做一次量化：
  原始 scale：FP16
  再量化 scale：3bit
```

###### 答主怎么说

> SpQR 将 LLMs 的内存占用减少了约 3.4 倍或更多，从而实现小于 1% 的精度损失。

---

###### ⑨ ZeroQuant

###### 原理

> 对权重做分组量化，对激活做逐 token 动态量化。不需要校准集。

###### 用我们数据推演

**权重**：分组量化（group=64 或 128，我们这里 group=2）

```
out0: [0.5, 0.1 | 0.3, 6.0]
out1: [0.4, 0.2 | 0.6, 0.8]
out2: [0.7, 0.3 | 0.5, 0.2]
out3: [0.1, 0.9 | 0.4, 0.6]
```

**激活**：逐 token 动态量化

```
推理时，拿到 X 马上算：
token0: [2.0, 0.5, 0.8, 0.3]
  → 动态算 s = 2.0/127 = 0.0157
  → 量化：2.0→127, 0.5→32, 0.8→51, 0.3→19

token1: [0.4, 7.0, 0.2, 0.6]
  → 动态算 s = 7.0/127 = 0.0551
  → 量化：0.4→7, 7.0→127, 0.2→4, 0.6→11
```

**Kernel 融合**：把量化操作融合到前一个算子（如 LayerNorm）中，减少数据移动开销。

---

###### ⑩ ZeroQuant-V2

###### 原理

> 新增两个低秩矩阵来补偿量化误差（LoRC）。

```
Y = X · W_int4_reconᵀ + X · (A · B)ᵀ
    ↑ 量化部分           ↑ 低秩补偿

A 和 B 很小，也可以量化为 8bit
```

###### 答主怎么说

> 该方式以最小的模型参数大小的增加来提升模型质量的恢复

---

###### ⑪ ZeroQuant-FP

###### 原理

> FP8 激活优于 INT8，FP4 权重和 INT4 相当。用位移（bit-shifting）代替反量化。

###### 用我们数据推演

**方法 M1**：把 scale 映射到最近的 2 的幂

```
原始 s = 0.0472
ŝ = 2^⌈log₂(0.0472)⌉ = 2^⌈-4.4⌉ = 2^(-4) = 0.0625

0.5 → round(0.5/0.0625) = 8  → 反量化 0.5   ✅ 更精确了！
0.1 → round(0.1/0.0625) = 2  → 反量化 0.125 ❌ 误差 0.025
0.3 → round(0.3/0.0625) = 5  → 反量化 0.3125 ❌ 误差 0.0125
6.0 → round(6.0/0.0625) = 96 → 反量化 6.0   ✅
```

**为什么用 2 的幂**？因为位移操作比浮点乘法快得多。

---

###### ⑫ FP6-LLM（W6A16）

###### 原理

> 没有原生 6bit 硬件支持，通过 TC-FPx kernel 让 GPU 支持 6bit 计算。

###### 答主怎么说

> 在批处理大小小于 32 的 LLM 推理任务中，性能比 FP16 基准高出 1.69 到 2.65 倍。

###### 限制

> 在 GEMM 因批处理较大或有充足的 GPU 内存而使得瓶颈变为 Tensor Core 计算时，可能无法保持其性能优势。

---

###### ⑬ KIVI

###### 原理

> KV 缓存 2bit 量化。Key 按通道量化，Value 按 token 量化。

###### 用我们数据推演

假设生成到第 3 步时 KV Cache：

```
Key 缓存（3×4）：
        c0   c1   c2   c3
step1: [0.5, 0.2, 0.8, 0.3]
step2: [0.4, 0.6, 0.2, 0.7]
step3: [0.8, 5.0, 0.3, 0.1]

Key 按通道量化（per-channel）：
  c0 列: [0.5, 0.4, 0.8] → s_c0 = 0.8/1 = 0.8
  c1 列: [0.2, 0.6, 5.0] → s_c1 = 5.0/1 = 5.0  ← 有离群值，步长粗
  c2 列: [0.8, 0.2, 0.3] → s_c2 = 0.8/1 = 0.8
  c3 列: [0.3, 0.7, 0.1] → s_c3 = 0.7/1 = 0.7

Value 缓存（3×4）：
        c0   c1   c2   c3
step1: [0.3, 0.7, 0.1, 0.5]
step2: [0.6, 0.2, 0.4, 0.8]
step3: [0.1, 0.4, 0.9, 0.2]

Value 按 token 量化（per-token）：
  step1: [0.3, 0.7, 0.1, 0.5] → s_step1 = 0.7/1 = 0.7
  step2: [0.6, 0.2, 0.4, 0.8] → s_step2 = 0.8/1 = 0.8
  step3: [0.1, 0.4, 0.9, 0.2] → s_step3 = 0.9/1 = 0.9
```

**为什么 Key 按通道，Value 按 token？**

Key 的某些通道（如 c1）在不同 token 间变化大，所以按通道量化可以独立处理"大通道"和"小通道"。

Value 的每行（每个 token）有自己的范围，所以按 token 量化更自然。

---

###### ⑭ IntactKV

###### 原理

> 保持关键词元的 KV 缓存无损（全精度）。

###### 用我们数据

```
生成序列："我 爱 吃 苹果"
第一个 token "我" 的 KV 缓存保留 FP16 不量化：
  K_<s> = [0.5, 0.2, 0.8, 0.3]  ← FP16
  V_<s> = [0.3, 0.7, 0.1, 0.5]  ← FP16

后面的 token 正常量化：
  K_爱 = INT4, K_吃 = INT4, K_苹果 = INT4
```

###### 答主怎么说

> 利用 LLM 推理时的公共前缀防止给量化模型的推理带来任何额外开销

---

###### ⑮ KVQuant

###### 原理

> 离线校准 + 多种优化技巧的组合。

###### 关键技巧

**① RoPE 之前量化 Key**

```
量化 Key 在 RoPE 之前做，因为 RoPE 会改变 Key 的分布，让量化更难。

量化前：K = [0.5, 0.2, 0.8, 0.3]  ← 范围小，容易量化
应用 RoPE：K_rotated = 旋转后的值     ← 范围变大，更难量化
```

**② Attention Sink 感知量化**

```
第一个 token 保留 FP16：
  K_<s> = [0.5, 0.2, 0.8, 0.3]  ← FP16，不量化
后面的 token 正常量化：
  K_step2 = INT4, K_step3 = INT4, ...
```

---

###### ⑯ Atom

###### 原理

> W4A4KV4，混合精度，异常值用 INT8。

###### 用我们数据

```
W 中离群值 6.0 → 用 INT8 存
W 中其他值 0.5, 0.1, 0.3 → 用 INT4 存

X 中离群值 7.0 → 用 INT8 存
X 中其他值 → 用 INT4 存
```

---

###### ⑰ QuaRot

###### 原理

> 用 Hadamard 旋转矩阵消除异常值，然后就可以用简单的 RTN 量化了。

###### 用我们数据

```
Hadamard 矩阵 H₂ = [1, 1; 1, -1] 对 2 维向量做旋转

X = [2.0, 0.5, 0.8, 0.3]
每两个元素做 Hadamard 变换：
  [2.0, 0.5] × H₂ = [2.5, 1.5]   ← 离群值被"稀释"了
  [0.8, 0.3] × H₂ = [1.1, 0.5]

旋转后 X' = [2.5, 1.5, 1.1, 0.5]  ← 没有明显的离群值了！
```

###### 答主怎么说

> 使用随机 Hadamard 变换旋转模型输入来解决异常特征问题

---

###### ⑱ QoQ

###### 原理

> W4A8KV4，渐进式分组量化 + SmoothAttention。

###### 渐进式分组量化

```
先对 W 做 per-channel INT8 量化
再对中间的 INT8 权重做 per-group INT4 量化

out0: [0.5, 0.1, 0.3, 6.0]
  Step 1: per-channel INT8 → s = 6.0/127 = 0.0472
          [11, 2, 6, 127]
  Step 2: per-group INT4 → 每组 2 个元素
          [11, 2] → s=11/7=1.57 → [7, 1]
          [6, 127] → s=127/7=18.1 → [0, 7]
```

###### SmoothAttention

> 受 SmoothQuant 启发，对 Key 的异常值通道做平滑

---

###### 总结：这些方法的核心思想

```
RTN:         四舍五入，最简单，精度最差
LLM-QAT:     训练时模拟量化，让模型提前适应
QLoRA:       量化后加 LoRA 补偿
LLM.int8():  INT8/FP16 混合精度，离群值单独处理
GPTQ:        量化后调整其他权重来补偿，优化视角
SmoothQuant: 平滑因子把量化难度从激活转移到权重
AWQ:         保护显著通道，按重要性分配精度
SpQR:        隔离离群权重 + 双层量化
ZeroQuant:   分组权重 + 逐 token 动态激活
ZeroQuant-V2:加低秩矩阵补偿误差
ZeroQuant-FP:FP4/FP8 + 位移加速
FP6-LLM:    TC-FPx kernel 支持 6bit
KIVI:        KV 2bit，Key 按通道/Value 按 token
IntactKV:    关键词元 KV 无损
KVQuant:     多种技巧组合优化 KV 量化
Atom:        W4A4KV4，混合精度处理异常值
QuaRot:      Hadamard 旋转消除异常值
QoQ:         W4A8KV4，渐进式分组量化
```

---

这 18 个方法每个都过了一遍，有哪个需要展开再详细讲吗？

## 2.1 RTN

**特点**：最简单的量化方法

RTN（Round to Nearest）量化是最简单的量化方法，就是采取四舍五入的方式，把量化权重近似到最近的整数上。对于 LLM 而言，由于大量异常值的存在，如果单独使用 RTN 量化，对精度影响较大。通常会采用一些方式抑制异常值之后，再使用 RTN 量化。

### RTN 的完整计算流程（INT4 对称量化）

```
第 1 步：确定量化参数（对 out0 行做 per-channel INT4 对称量化）
  out0: [0.5, 0.1, 0.3, 6.0]
  INT4 范围 -8~7，z=0
  s = max(|0.5|,|0.1|,|0.3|,|6.0|) / 7 = 6.0/7 ≈ 0.8571

第 2 步：逐个 round
  0.5 → round(0.5/0.8571)=1   → 存为 1
  0.1 → round(0.1/0.8571)=0   → 存为 0
  0.3 → round(0.3/0.8571)=0   → 存为 0
  6.0 → round(6.0/0.8571)=7   → 存为 7 ✅
  out0 存为 INT4: [1, 0, 0, 7]  ← 只用 16bit（原来 64bit）

第 3 步：反量化
  1 → 1×0.8571 = 0.857   原值 0.5  误差 +0.357
  0 → 0                   原值 0.1  误差 -0.1
  0 → 0                   原值 0.3  误差 -0.3
  7 → 6.0                 原值 6.0  误差 0
  W_recon_out0 = [0.857, 0, 0, 6.0]

第 4 步：用反量化后的权重算 Y
  Y[0][0] = 2.0×0.857 + 0.5×0 + 0.8×0 + 0.3×6.0 = 3.514
  原始 FP16: Y[0][0] = 2.0×0.5 + 0.5×0.1 + 0.8×0.3 + 0.3×6.0 = 3.09
  相对误差 = 0.424/3.09 = 13.7%  ❌
```

**小值全被"吞"成 0 了**——一个 `6.0` 让三个小值的有效分辨率变得很差。

### 对比不同粒度的 RTN

**① Per-tensor RTN（整个矩阵一个 scale）**：s=6.0/7=0.8571，所有行都用这个粗 scale，很多小值（0.1,0.2,0.3）都被 round 成 0。

**② Per-channel RTN（每行一个 scale）**：

```
out0: s₀=6.0/7=0.8571 → [0.857, 0, 0, 6.0]   误差大（被 6.0 拖累）
out1: s₁=0.8/7=0.1143 → [0.571, 0.229, 0.571, 0.800]  ✅ 精度很好
out2: s₂=0.7/7=0.1    → [0.7, 0.3, 0.5, 0.2]          ✅ 误差极小
out3: s₃=0.9/7=0.1286 → [0.129, 0.900, 0.386, 0.643]  ✅
结论：只有 out0 被 6.0 拖累，其他行精度都很好。
```

**③ Per-group RTN（每行分成 2 组）**：

```
out0: [0.5, 0.1 | 0.3, 6.0]
组A₀: s=0.5/7=0.0714 → 0.5→7→0.5 ✅, 0.1→1→0.071 ❌(误差-0.029)
组B₀: s=6.0/7=0.8571 → 0.3→0→0 ❌(误差-0.3), 6.0→7→6.0 ✅

对比 per-channel：0.5 从 0.857(误差0.357) 变回 0.5(误差0)，
               0.1 从 0(误差-0.1) 变回 0.071(误差-0.029)
per-group 把 0.5 和 0.1 从"死亡的边缘"救了回来！
```

**如果 group 缩到 1（per-element）**：每个元素独立一个 scale，误差=0。但 **per-element 没有压缩效果**（每个元素都存了一个 scale），实际中不会用。

### RTN 的三种误差来源总结

| 误差类型 | 在我们的数据中 | 根本原因 |
|:--------|:-------------|:--------|
| **截断误差（Clip）** | 6.0/0.8571=7.0 刚好在边界，没截断；但如果有个 6.5 就会被截断到 7 | 值超出量化范围 |
| **舍入误差（Round）** | 0.5→1→0.857（误差 0.357），0.1→0→0（误差 -0.1） | 四舍五入的精度损失 |
| **粒度误差（Granularity）** | out0 的 0.5 被 6.0 拖累，整行的 s 被拉大 | 共用 scale 的粒度太粗 |

**RTN 改进的路径就是三条**：
- 减少舍入误差 → 用更细的粒度（per-group）
- 减少截断误差 → 动态量化（runtime 统计当前范围）
- 抑制异常值 → SmoothQuant/AWQ 的数学变换

> ⚠️ **易错校正**：RTN = Round To Nearest，`q = round(w/scale)`。它没有"理解模型"，也不利用 Hessian 或 activation importance。**后面很多方法本质上都是在回答：如何先处理 outlier / sensitivity，再让简单 RTN 也能工作？**

---

### 📌 原始 Q&A：RTN（历史第 36 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 36 轮）

好了现在是这个：RTN

##### A（历史第 37 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！我们深入推演 **RTN（Round to Nearest）**，用 4×4 数据跑一遍完整过程。

---

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

激活 X(2×4)：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]
```

---

###### RTN 的核心思想

**四个字：四舍五入（Round to Nearest）**

```
量化： q = round(r / s) + z
反量化： r_recon = (q - z) × s
```

RTN 是所有量化方法中最简单的，它不做任何"优化"——直接四舍五入到最近的整数格子。

---

###### 一、RTN 的完整计算流程（INT4 对称量化）

###### 第 1 步：确定量化参数

对 out0 行做 **per-channel INT4 对称量化**：

```
out0: [0.5, 0.1, 0.3, 6.0]

INT4 对称量化（范围 -8 ~ 7，z=0）：
s = max(|0.5|, |0.1|, |0.3|, |6.0|) / 7
  = 6.0 / 7
  ≈ 0.8571
```

###### 第 2 步：逐个 round

```
0.5 → round(0.5 / 0.8571) = round(0.583) = 1   → 存为 1
0.1 → round(0.1 / 0.8571) = round(0.117) = 0   → 存为 0
0.3 → round(0.3 / 0.8571) = round(0.350) = 0   → 存为 0
6.0 → round(6.0 / 0.8571) = round(7.000) = 7   → 存为 7 ✅

out0 存为 INT4: [1, 0, 0, 7]   ← 只用了 16bit（原来是 64bit FP16）
```

###### 第 3 步：反量化

```
1 → 1 × 0.8571 = 0.857  原值 0.5  误差 +0.357
0 → 0 × 0.8571 = 0      原值 0.1  误差 -0.1
0 → 0 × 0.8571 = 0      原值 0.3  误差 -0.3
7 → 7 × 0.8571 = 6.0    原值 6.0  误差 0

W_recon_out0 = [0.857, 0, 0, 6.0]
```

###### 第 4 步：用反量化后的权重算 Y

```
Y[0][0] = token0 · W_recon_out0
        = 2.0×0.857 + 0.5×0 + 0.8×0 + 0.3×6.0
        = 1.714 + 0 + 0 + 1.8
        = 3.514

原始 FP16: Y[0][0] = token0 · W_out0
           = 2.0×0.5 + 0.5×0.1 + 0.8×0.3 + 0.3×6.0
           = 1.0 + 0.05 + 0.24 + 1.8
           = 3.09

误差 = 3.514 - 3.09 = 0.424  ❌
相对误差 = 0.424/3.09 = 13.7%
```

---

###### 二、对比不同粒度的 RTN

用同一份数据，看 **per-tensor** vs **per-channel** vs **per-group** 的区别。

###### 1. Per-tensor RTN（整个矩阵一个 scale）

```
W 所有 16 个值：
[0.5, 0.1, 0.3, 6.0, 0.4, 0.2, 0.6, 0.8, 0.7, 0.3, 0.5, 0.2, 0.1, 0.9, 0.4, 0.6]

max(|W|) = 6.0

s = 6.0 / 7 = 0.8571

每个值都除以 0.8571 后 round：

out0: [0.5→1, 0.1→0, 0.3→0, 6.0→7]   → 反量化 [0.857, 0, 0, 6.0]
out1: [0.4→0, 0.2→0, 0.6→1, 0.8→1]   → 反量化 [0, 0, 0.857, 0.857]
out2: [0.7→1, 0.3→0, 0.5→1, 0.2→0]   → 反量化 [0.857, 0, 0.857, 0]
out3: [0.1→0, 0.9→1, 0.4→0, 0.6→1]   → 反量化 [0, 0.857, 0, 0.857]

问题：很多小值（0.1, 0.2, 0.3）都被 round 成 0 了！
```

###### 2. Per-channel RTN（每行一个 scale）

```
out0: s₀ = 6.0/7 = 0.8571 → [0.857, 0, 0, 6.0]      误差大
out1: s₁ = 0.8/7 = 0.1143 → [0.571, 0.229, 0.571, 0.800]
      ↑ s₁ 很小，out1 的精度很好！
      0.4→round(0.4/0.1143)=3→3×0.1143=0.343  误差-0.057 ✅
      0.2→round(0.2/0.1143)=2→2×0.1143=0.229  误差+0.029 ✅
      0.6→round(0.6/0.1143)=5→5×0.1143=0.571  误差-0.029 ✅
      0.8→round(0.8/0.1143)=7→7×0.1143=0.800  误差0     ✅

out2: s₂ = 0.7/7 = 0.1     → [0.7, 0.3, 0.5, 0.2]     ✅ 误差极小
out3: s₃ = 0.9/7 = 0.1286  → [0.129, 0.900, 0.386, 0.643] ✅

结论：只有 out0 被 6.0 拖累，其他行精度都很好。
```

###### 3. Per-group RTN（每行分成 2 组）

```
out0: [0.5, 0.1 | 0.3, 6.0]

组A₀: s = 0.5/7 = 0.0714
  0.5→round(0.5/0.0714)=7→7→0.5   ✅
  0.1→round(0.1/0.0714)=1→1→0.071 ❌ 误差 -0.029 ← 比 per-channel 的 0 好多了！

组B₀: s = 6.0/7 = 0.8571
  0.3→round(0.3/0.8571)=0→0→0     ❌ 误差 -0.3
  6.0→round(6.0/0.8571)=7→7→6.0   ✅
```

**对比 per-channel vs per-group 对 out0 的修复效果**：

```
per-channel out0: [0.857, 0, 0, 6.0]
                 0.5→0.857(误差0.357), 0.1→0(误差-0.1)

per-group out0:   [0.5, 0.071, 0, 6.0]
                 0.5→0.5(误差0), 0.1→0.071(误差-0.029)

per-group 把 0.5 和 0.1 从死亡的边缘救了回来！
```

---

###### 三、per-group 粒度下看 6.0 对组内的影响

per-group 虽然把 0.5 和 0.1 救了回来，但 **组 B₀ 的 0.3 还是被 6.0 害了**。

```
组B₀: [0.3, 6.0]
s = 6.0/7 = 0.8571

0.3 → round(0.3/0.8571) = round(0.35) = 0 → 反量化 0
```

如果我把 group size 从 2 缩小到 1（即每个元素独立量化）：

```
每个元素单独一个 scale：
  0.3: s=0.3/7=0.0429 → round(0.3/0.0429)=7 → 反量化 0.3 ✅
  6.0: s=6.0/7=0.8571 → round(6.0/0.8571)=7 → 反量化 6.0 ✅

误差 = 0！
```

但 **per-element 没有压缩效果**（每个元素都存了一个 scale），所以实际中不会用。

---

###### 四、答主原话怎么说的

> **RTN 量化是最简单的量化方法，就是采取四舍五入的方式，把量化权重近似到最近的整数上。**

> **对于LLM而言，由于大量异常值的存在，如果单独使用RTN量化，对精度影响较大。通常会采用一些方式抑制异常值之后，再使用RTN量化。**

"抑制异常值之后，再使用RTN量化"——这就是 SmoothQuant 和 AWQ 做的事：

```
SmoothQuant: 用平滑因子 s 把异常值"压"下去 → 再 RTN
AWQ: 用缩放因子 α 保护显著通道 → 再 RTN
```

---

###### 五、RTN 的误差来源总结

用我们数据归纳 RTN 的三种误差：

| 误差类型 | 在我们的数据中 | 根本原因 |
|:--------|:-------------|:--------|
| **截断误差（Clip）** | 6.0/0.8571=7.0，刚好在边界，没截断。但如果有个 6.5 就会被截断到 7 | 值超出量化范围 |
| **舍入误差（Round）** | 0.5→1→0.857（误差 0.357），0.1→0→0（误差 -0.1） | 四舍五入的精度损失 |
| **粒度误差（Granularity）** | out0 的 0.5 被 6.0 拖累，整行的 s 被拉大 | 共用 scale 的粒度太粗 |

**RTN 改进的路径就是三条**：
- 减少舍入误差 → 用更细的粒度（per-group）
- 减少截断误差 → 动态量化（runtime 统计当前范围）
- 抑制异常值 → SmoothQuant/AWQ 的数学变换

---

需要我像这样继续推演 **LLM-QAT**、**QLoRA** 或其他方法吗？

## 2.2 LLM-QAT

**特点**：训练感知量化、无数据知识蒸馏、对称 MinMax 均匀量化（线性量化）、逐通道权重量化，逐 Token 激活量化，逐 Token KV缓存量化

大语言模型中，权重和激活都存在显著的异常值。这些异常值对量化过程有显著影响，因为它们会增加量化步长，同时降低中间值的精度。因此，在量化过程中裁剪这些异常值不利于 LLM 的性能。

在训练的初始阶段，任何基于裁剪的方法都会导致异常高的困惑度，从而丢失大量信息，并且通过事实证明很难通过微调来恢复。因此，选择保留这些异常值。

此外，在具有门控线性单元（GLU）的模型中，激活权重大多是对称分布的。因此，为权重和激活选择对称 MinMax 量化，对权重采用逐通道量化，对激活采用逐 Token 量化。

同时，大语言模型推理工程中的 KV 缓存也会消耗不少的内存，对于 KV 缓存进行逐 Token 量化，Key 和 Value 是由 Token 生成的。在生成过程中，当前的 Key 和 Value 都会被量化，并存储它们对应的缩放因子。在 QAT 的训练过程中，对 key 和 value 的整个激活张量进行量化。并通过将量化函数集成到梯度计算中，确保使用量化的键值对进行有效的训练。

> 🔍 **学习追问：LLM-QAT 是训练的还是推理的？**
>
> **LLM-QAT 是训练阶段的（QAT）。** 它在模型训练/预训练阶段就模拟量化，让模型提前适应低精度。与 PTQ 不同，QAT 需要完整的训练流程和计算资源，但精度更高。

### LLM-QAT 的量化策略

- 量化类型：**对称 MinMax**（z=0）
- 权重粒度：**per-channel**
- 激活粒度：**per-token**
- KV 缓存粒度：**per-token**
- 关键设计：**不裁剪异常值**（裁剪会导致困惑度飙升且难以恢复）

### 用数据推演 LLM-QAT 的完整流程

**第 1 步：前向传播 — 插入伪量化节点**

```
正常训练：W_fp16 → Y = X·Wᵀ → Loss
QAT 训练：W_fp16 → 伪量化 → W_fake_int8 → 反量化 → W_recon_fp16
          X_fp16 → 伪量化 → X_fake_int8 → 反量化 → X_recon_fp16
          Y = X_recon · W_reconᵀ → Loss
```

**伪量化（Fake Quantization）**：只是"模拟"量化，不是真的存成 INT8。它把 FP16 值压到 INT8 再解压回来，让模型提前感受到量化误差。

**第 2 步：权重的伪量化（per-channel 对称）**

```
out0: [0.5, 0.1, 0.3, 6.0]
s₀ = 6.0/127 = 0.0472
  0.5 → round(0.5/0.0472)=11 → 11×0.0472=0.519  ← 误差 +0.019
  0.1 → round(0.1/0.0472)=2  → 2×0.0472=0.094   ← 误差 -0.006
  0.3 → round(0.3/0.0472)=6  → 6×0.0472=0.283   ← 误差 -0.017
  6.0 → round(6.0/0.0472)=127 → 127×0.0472=6.0  ✅
W_recon_out0 = [0.519, 0.094, 0.283, 6.0]

out1: [0.4, 0.2, 0.6, 0.8]
s₁ = 0.8/127 = 0.0063
  0.4→63→0.397  ← 误差 -0.003 ✅
  0.2→32→0.202  ← 误差 +0.002 ✅
  0.6→95→0.599  ← 误差 -0.001 ✅
  0.8→127→0.800 ✅
```

**第 3 步：激活的伪量化（per-token 对称）**

```
token0: [2.0, 0.5, 0.8, 0.3]
s_token0 = 2.0/127 = 0.0157
  2.0→127→2.0   ✅
  0.5→32→0.502  ← 误差 +0.002 ✅
  0.8→51→0.801  ← 误差 +0.001 ✅
  0.3→19→0.298  ← 误差 -0.002 ✅
X_recon_token0 = [2.0, 0.502, 0.801, 0.298]
```

**第 4 步：用伪量化的值算 Y、算 Loss**

```
Y_recon[0][0] = 2.0×0.519 + 0.502×0.094 + 0.801×0.283 + 0.298×6.0 = 3.100
原始 Y[0][0] = 3.09
误差 = 0.010  ← 很小 ✅（因为保留了异常值，没有裁剪）
```

**第 5 步：反向传播更新 W**

```
伪量化节点本身不可微（round 函数梯度为 0），但 QAT 用直通估计器（STE）让梯度直接穿透它：
∂Loss/∂W_recon → 直接等于 → ∂Loss/∂W_original
W_new = W_old - lr × ∂Loss/∂W   ← 更新的是原始 FP16 的 W
```

**第 6 步：训练完成后，真正量化部署**

```
训练后的 W（已被微调）：out0: [0.48, 0.12, 0.28, 5.9]  ← 相比原始 [0.5,0.1,0.3,6.0] 被微调了
对这个 W 做真正的 INT8 量化部署。

对比没有 LLM-QAT vs 有 LLM-QAT：
  没有（直接 PTQ 量化 out0）：Y[0][0]=3.100  误差 0.010
  有（训练后的 W 再量化）：   Y[0][0]=3.092  误差 0.002 ← 精度更高！
```

### 无数据知识蒸馏

> **LLM-QAT 不需要原始训练数据**，而是用模型自己生成的数据来做蒸馏。
>
> ```
> Teacher 模型（全精度 LLM）→ 生成输出分布
> Student 模型（量化 LLM）→ 学习模仿 Teacher 的输出分布
> Loss 不是和真实标签比，而是和 Teacher 的输出比。
> ```

> 🔍 **LLM-QAT 总结（训练 vs 推理）**
>
> | 阶段 | LLM-QAT 在做什么 |
> |:----|:----------------|
> | **训练阶段** | 插入伪量化节点，模拟量化误差，通过反向传播微调 W，让 W 学会适应低精度 |
> | **推理阶段** | W 已经被微调好了，直接做真正的 INT8 量化部署，精度比 PTQ 好 |

---

**博主原文配图**（源自知乎原文）：

![原文图 4](https://picx.zhimg.com/v2-26706ae233f968e014ba0c973e06ca44_r.jpg?source=2c26e567)


![原文图 5](https://picx.zhimg.com/v2-69b4dec6c5b3562243ff256156198325_r.jpg?source=2c26e567)


---

### 📌 原始 Q&A：LLM-QAT（历史第 38 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 38 轮）

LLM-QAT
这部分内容呢？这部分是 训练的还是推理的？

##### A（历史第 39 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！先回答你的问题，然后推演。

###### 先回答：LLM-QAT 是 **训练阶段** 的

答主原文的分类很清楚：

> **量化感知训练（QAT）**：在模型训练过程中加入伪量化算子

LLM-QAT 的全称是 **LLM 版的 QAT**（Quantization Aware Training）。它在训练/预训练阶段就模拟量化，让模型提前适应低精度。

和 PTQ 的对比：

| | PTQ（如 GPTQ/AWQ） | LLM-QAT |
|:--|:------------------|:--------|
| **阶段** | 训练完成后 | 训练过程中 |
| **反向传播** | 不需要 | 需要 |
| **数据需求** | 少量校准数据 | 大量训练数据（或知识蒸馏） |
| **是否改权重** | 不动原始 W（或微调补偿） | 训练过程中更新 W |

---

###### LLM-QAT 的完整推演

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

激活 X(2×4)：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]
```

---

###### 第 1 步：LLM-QAT 的量化策略选择

答主原文说：

> 在具有门控线性单元（GLU）的模型中，激活权重大多是对称分布的。因此，为权重和激活选择对称 MinMax 量化。

**LLM-QAT 的量化策略**：
- 量化类型：**对称 MinMax**（z=0）
- 权重粒度：**per-channel**
- 激活粒度：**per-token**
- KV 缓存粒度：**per-token**

---

###### 第 2 步：前向传播 — 插入伪量化节点

正常训练的前向：

```
W_fp16 → Y = X · W_fp16ᵀ → Loss
```

LLM-QAT 的前向（插入伪量化）：

```
W_fp16 → 伪量化 → W_fake_int8 → 反量化 → W_recon_fp16
X_fp16 → 伪量化 → X_fake_int8 → 反量化 → X_recon_fp16

Y = X_recon · W_reconᵀ → Loss
```

**伪量化（Fake Quantization）**：只是"模拟"量化，不是真的存成 INT8。它把 FP16 值压到 INT8 再解压回来，让模型提前感受到量化误差。

---

###### 第 3 步：用我们数据走一遍伪量化

**权重的伪量化（per-channel 对称）**：

```
out0: [0.5, 0.1, 0.3, 6.0]
s₀ = max(0.5, 0.1, 0.3, 6.0) / 127 = 6.0/127 = 0.0472

0.5 → round(0.5/0.0472) = round(10.6) = 11 → 11×0.0472 = 0.519
0.1 → round(0.1/0.0472) = round(2.1)  = 2  → 2×0.0472 = 0.094
0.3 → round(0.3/0.0472) = round(6.4)  = 6  → 6×0.0472 = 0.283
6.0 → round(6.0/0.0472) = round(127)  = 127 → 127×0.0472 = 6.0

W_recon_out0 = [0.519, 0.094, 0.283, 6.0]

out1: [0.4, 0.2, 0.6, 0.8]
s₁ = 0.8/127 = 0.0063

0.4 → round(0.4/0.0063)=63 → 63×0.0063=0.397  ← 误差 -0.003 ✅
0.2 → round(0.2/0.0063)=32 → 32×0.0063=0.202  ← 误差 +0.002 ✅
0.6 → round(0.6/0.0063)=95 → 95×0.0063=0.599  ← 误差 -0.001 ✅
0.8 → round(0.8/0.0063)=127 → 127×0.0063=0.800 ✅
```

**激活的伪量化（per-token 对称）**：

```
token0: [2.0, 0.5, 0.8, 0.3]
s_token0 = 2.0/127 = 0.0157

2.0 → round(2.0/0.0157)=127 → 127×0.0157=2.0   ✅
0.5 → round(0.5/0.0157)=32  → 32×0.0157=0.502  ← 误差 +0.002 ✅
0.8 → round(0.8/0.0157)=51  → 51×0.0157=0.801  ← 误差 +0.001 ✅
0.3 → round(0.3/0.0157)=19  → 19×0.0157=0.298  ← 误差 -0.002 ✅

X_recon_token0 = [2.0, 0.502, 0.801, 0.298]

token1: [0.4, 7.0, 0.2, 0.6]
s_token1 = 7.0/127 = 0.0551

0.4 → round(0.4/0.0551)=7  → 7×0.0551=0.386   ← 误差 -0.014
7.0 → round(7.0/0.0551)=127 → 127×0.0551=7.0   ✅
0.2 → round(0.2/0.0551)=4  → 4×0.0551=0.220   ← 误差 +0.020
0.6 → round(0.6/0.0551)=11 → 11×0.0551=0.606   ← 误差 +0.006

X_recon_token1 = [0.386, 7.0, 0.220, 0.606]
```

---

###### 第 4 步：用伪量化的值计算 Y，算 Loss

```
Y_recon = X_recon · W_reconᵀ

Y_recon[0][0] = token0_recon · out0_recon
  = 2.0×0.519 + 0.502×0.094 + 0.801×0.283 + 0.298×6.0
  = 1.038 + 0.047 + 0.227 + 1.788
  = 3.100

原始 Y[0][0] = 3.09
误差 = 3.100 - 3.09 = 0.010 ← 误差很小 ✅
```

**看，LLM-QAT 因为保留了异常值（没有裁剪），所以精度损失很小。**

答主原文原话：

> 在训练的初始阶段，任何基于裁剪的方法都会导致异常高的困惑度...因此，选择保留这些异常值。

---

###### 第 5 步：反向传播更新 W

Loss 算出来后，反向传播，**梯度穿透伪量化节点**。

伪量化节点本身不可微（round 函数梯度为 0），但 QAT 的 **直通估计器（STE）** 让梯度直接穿透它：

```
∂Loss/∂W_recon → 直接等于 → ∂Loss/∂W_original

W_new = W_old - lr × ∂Loss/∂W
```

所以更新的是 **原始 FP16 的 W**，不是被量化后的 W_recon。

```
假设学习率 lr=0.01，梯度 ∂Loss/∂W_out0_c0=0.1：

W_new_out0_c0 = 0.5 - 0.01 × 0.1 = 0.499

经过多轮训练后，W 会逐渐演化成：
  "即使被量化到 INT8 再解压回来，输出误差也最小" 的状态
```

---

###### 第 6 步：训练完成后，真正量化部署

LLM-QAT 训练结束后，得到的是 **已经被微调过的全精度 W**：

```
训练结束时的 W：
out0: [0.48, 0.12, 0.28, 5.9]  ← 相比原始 [0.5, 0.1, 0.3, 6.0] 被微调了
out1: [0.39, 0.21, 0.58, 0.81]
out2: [0.68, 0.31, 0.49, 0.21]
out3: [0.11, 0.88, 0.41, 0.62]
```

然后对这个 W 做 **真正的 INT8 量化**，部署推理。

因为 W 已经在训练过程中适应了量化误差，所以 **量化后的精度比没有经过 QAT 的 W 好得多**。

---

###### 对比：没有 LLM-QAT vs 有 LLM-QAT

```
没有 LLM-QAT（直接 PTQ 量化 out0）：
  原始 W:     [0.5, 0.1, 0.3, 6.0]
  量化后:     [0.519, 0.094, 0.283, 6.0]
  Y[0][0] = 3.100  误差 0.010

有 LLM-QAT（训练后的 W 再量化）：
  训练后 W:   [0.48, 0.12, 0.28, 5.9]  ← 被微调过了
  量化后:     [0.479, 0.126, 0.283, 5.9]
  Y[0][0] = 3.092  误差 0.002 ← 精度更高！
```

---

###### 答主还提到了 LLM-QAT 的另一个特点：无数据知识蒸馏

> **无数据知识蒸馏**：LLM-QAT 不需要原始训练数据，而是用模型自己生成的数据来做蒸馏。

```
Teacher 模型（全精度 LLM）→ 生成输出分布
Student 模型（量化 LLM）→ 学习模仿 Teacher 的输出分布

Loss 不是和真实标签比，而是和 Teacher 的输出比。
```

这样即使拿不到原始训练数据，也能做 QAT。

---

###### 一张图总结 LLM-QAT

```
训练阶段（LLM-QAT）：
  每步前向：
    W_fp16 → 伪量化 → W_recon_fp16  (模拟 INT8 的精度损失)
    X_fp16 → 伪量化 → X_recon_fp16  (模拟 INT8 的精度损失)
    Y = X_recon · W_reconᵀ
    Loss = MSE(Y, Y_teacher)  (知识蒸馏)
  
  反向传播：
    梯度穿透伪量化节点 (STE)
    更新 W_fp16 ← 这是关键！W 在训练中被微调

训练完成后：
  得到微调过的 W_fp16
  做真正的 INT8 量化 → 部署

推理阶段（真正部署）：
  W_int8 + X_int8 → INT8 GEMM → 反量化 → Y_fp16
  此时 W 已经"适应"了 INT8 的精度损失，所以效果好
```

---

所以回到你的问题：

| 阶段 | LLM-QAT 在做什么 |
|:----|:----------------|
| **训练阶段** | 插入伪量化节点，模拟量化误差，通过反向传播微调 W，让 W 学会适应低精度 |
| **推理阶段** | W 已经被微调好了，直接做真正的 INT8 量化部署，精度比 PTQ 好 |

需要继续推演 **QLoRA** 吗？

## 2.3 QLoRA

**特点**：量化感知微调、分页优化器、双量化

QLoRA 将预训练模型量化为 4 比特，然后添加一小组可学习的 LoRA 权重，这些权重通过量化权重的反向传播梯度进行微调。无论何时使用 QLoRA 权重张量，都会将张量反量化为 BFloat16，然后执行 16 位矩阵乘法。

在 QLoRA 中，提出了两种技术实现来保证 4 比特微调的精度——**NF4 量化**和**双量化**。[NF4数据类型](https://zhida.zhihu.com/search?content_id=710333991&content_type=Answer&match_order=1&q=NF4%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B&zhida_source=entity) 对正态分布数据比 INT4/FP4 更优。双量化对第一次量化后的那些常量再进行一次量化，进一步减少存储空间。

此外，还引入了**分页优化器**，以防止梯度检查点期间的内存峰值，从而导致内存不足的错误。使用此功能为优化器状态（Optimizer）分配分页内存，然后在 GPU 内存不足时将其自动卸载到 CPU 内存，并在优化器更新需要时将其加载回 GPU 内存。

> 🔍 **先定位：QLoRA 是哪个阶段？**
>
> QLoRA 属于 **QAF（量化感知微调）**，不是 QAT（全量训练），也不是 PTQ（训练后直接量化）。
>
> | | PTQ | QAF（QLoRA） | QAT（LLM-QAT） |
> |:--|:---|:------------|:-------------|
> | 什么时候做 | 训练完成后 | **微调阶段** | 训练过程中 |
> | 数据量 | 少量校准数据 | 微调数据（相对少） | 全量训练数据 |
> | 是否改 W | 不动，或微调补偿 | **不动 W，加 LoRA 补偿** | 训练中更新 W |
> | 典型场景 | 快速部署 | 在特定任务上微调量化模型 | 从头训练量化友好模型 |

### 第 1 步：把 W 量化到 NF4（4bit）

**NF4 是什么？** 一种**非线性量化格式**，专为正态分布设计，16 个格子（4bit）中间密、两边疏：

```
NF4 的 16 个值：
索引:  0      1      2      3      4      5      6      7      8
值:   -1.0  -0.696 -0.525 -0.393 -0.277 -0.174 -0.085 -0.008 0.008
索引:  9      10     11     12     13     14     15
值:    0.085  0.174  0.277  0.393  0.525  0.696  1.0
格子间距：-0.008~0.008=0.016（密）  0.696~1.0=0.304（疏）
```

**用 NF4 量化 out0 行**：

```
out0: [0.5, 0.1, 0.3, 6.0]
absmax = 6.0
归一化：W_norm = W/absmax = [0.0833, 0.0167, 0.05, 1.0]
在 NF4 16 个值里找最近邻：
  0.0833 → 离 0.085(索引9) 最近 → 0.085
  0.0167 → 离 0.008(索引8) 最近 → 0.008
  0.05   → 离 0.085(索引9) 最近 → 0.085
  1.0    → 离 1.0(索引15) 最近 → 1.0
反量化还原（×absmax=6.0）：W_recon = [0.51, 0.048, 0.51, 6.0]

对比 INT4 和 NF4：
  原始值:  [0.5,  0.1,  0.3,  6.0]
  INT4:   [0.857, 0,   0,   6.0] ❌ 小值全死
  NF4:    [0.51, 0.048, 0.51, 6.0] ✅ 小值保住了
```

这就是 NF4 的优势——中间格子的密度大，小值的精度比 INT4 好得多。

### 第 2 步：冻结 W_int4，添加 LoRA

```
W_int4 冻结不动。
添加 LoRA（对 4×4 的 W，输出4维输入4维，rank=2）：
  W = W_int4_recon + LoRA_A × LoRA_B
  LoRA_A: 4×2（降维），LoRA_B: 2×4（升维）
  LoRA 参数量：4×2+2×4=16；原始 W 参数量：4×4=16
  但在大模型中：d=4096, r=8
    W=4096×4096=16M；LoRA=4096×8+8×4096=65K
    LoRA 只有 W 的 0.4% ← 这才是 QLoRA 的典型场景
```

### 第 3 步：前向传播

```
量化部分：
  W_int4 → 反量化 → W_fp16_recon = [0.51, 0.048, 0.51, 6.0]
  Y_quant[0][0] = 2.0×0.51+0.5×0.048+0.8×0.51+0.3×6.0 = 3.252
  原始 FP16 Y[0][0] = 3.09，误差 0.162

LoRA 补偿部分：
  LoRA_B 初始化为 0 → 初始时 Y_lora=0
  Y_total = Y_quant + 0 = 3.252
Loss = MSE(Y_total, Y_true)
```

### 第 4 步：反向传播 — 只更新 LoRA

```
∂Loss/∂W_int4 = 0（冻结，不更新）
∂Loss/∂LoRA_A、∂Loss/∂LoRA_B ← 更新
LoRA 学会补偿量化误差：Y_total = 3.252 + Y_lora → 更接近 3.09
```

### 第 5 步：部署推理

```
部署模型包含：1. W_int4 + scale 元数据（NF4）；2. LoRA_A + LoRA_B（FP16）
前向：W_int4 → 反量化 → W_fp16_recon
      Y = X·W_fp16_reconᵀ + X·(LoRA_A·LoRA_B)ᵀ
```

### 双量化（Double Quantization）

```
W 量化后，每组的 scale 也要存：
  假设 group=64，每 64 个权重共享一个 FP32 scale
  额外开销 = 32bit/64个参数 = 0.5 bit/参数
双量化：把 scale 也量化，比如 FP32→FP8
  额外开销 = 8bit/64个参数 = 0.125 bit/参数
  节省了 0.375 bit/参数！
```

### 分页优化器（Paged Optimizers）

```
微调时优化器状态（Adam 动量、方差）占大量显存。
QLoRA 分页优化器：放不下的优化器状态自动卸载到 CPU，需要时加载回 GPU（类似操作系统虚拟内存）。
```

> 🔍 **博主原文背景**：分页优化器用于**防止梯度检查点期间的内存峰值，从而导致内存不足（OOM）的错误**；在过去，这些错误使得大模型难以在单台机器上进行微调。使用此功能为优化器状态（Optimizer）分配分页内存，然后在 GPU 内存不足时将其自动卸载到 CPU 内存，并在优化器更新需要时将其加载回 GPU 内存。

### 整体对比：QLoRA vs LLM-QAT vs GPTQ

用 out0 行的 [0.5, 0.1, 0.3, 6.0] 看三种方法：

```
GPTQ（PTQ）：    W 被改为 [0.5, 0.15, 0.3, 6.0]（补偿调整），量化后 [0.857, 0, 0, 6.0]，只需几分钟
LLM-QAT（QAT）： 训练中 W 被微调为 [0.48, 0.12, 0.28, 5.9]，量化后 [0.479, 0.126, 0.283, 5.9]，需全量训练
QLoRA（QAF）：    W 不动 [0.5, 0.1, 0.3, 6.0]，NF4 量化后 [0.51, 0.048, 0.51, 6.0]，额外 LoRA 补偿，需少量微调数据
```

> ⚠️ **易错校正：QLoRA 的核心目标**
>
> QLoRA 的主要目标是**让大模型能够以低显存成本进行高质量微调**，它不是一个单纯为了 inference latency 的"推理量化算法"。LoRA 训练也不专门以"恢复量化前输出"为唯一目标，而是学习下游微调任务；它可能同时吸收一部分量化误差，但不能把 QLoRA 简化成"LoRA 专门补偿量化误差"。

---

### 📌 原始 Q&A：QLoRA（历史第 40 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 40 轮）

QLoRA 这部分呢？

##### A（历史第 41 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！用我们的 4×4 数据推演 **QLoRA**。

---

###### 先定位：QLoRA 是什么阶段？

答主原文的分类：

> **量化感知微调（QAF）**：在微调过程中对LLM进行量化。

QLoRA 属于 **QAF（量化感知微调）**，不是 QAT（全量训练），也不是 PTQ（训练后直接量化）。

**三个阶段的定位**：

| | PTQ | QAF（QLoRA） | QAT（LLM-QAT） |
|:--|:---|:------------|:-------------|
| 什么时候做 | 训练完成后 | **微调阶段** | 训练过程中 |
| 数据量 | 少量校准数据 | 微调数据（相对少） | 全量训练数据 |
| 是否改 W | 不动，或微调补偿 | **不动 W，加 LoRA 补偿** | 训练中更新 W |
| 典型场景 | 快速部署 | 在特定任务上微调量化模型 | 从头训练量化友好模型 |

**QLoRA 的核心思想**：预训练模型量化为 4bit 然后**冻结**，加一小组可学习的 LoRA 权重来补偿量化误差。

---

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

激活 X(2×4)：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]
```

---

###### 一、第 1 步：把 W 量化到 NF4（4bit）

###### NF4 是什么？

答主原文：

> **NF4 数据类型**：对正态分布数据比 INT4/FP4 更优。

NF4 是一种**非线性量化格式**，专为正态分布设计。它有 16 个格子（4bit），但格子不是均匀的——中间密、两边疏。

```
NF4 的 16 个值：
索引:  0      1      2      3      4      5      6      7      8
值:   -1.0  -0.696 -0.525 -0.393 -0.277 -0.174 -0.085 -0.008 0.008

索引:  9      10     11     12     13     14     15
值:    0.085  0.174  0.277  0.393  0.525  0.696  1.0

格子间距：
  -0.008 ~ 0.008 = 0.016  ← 很密（值密集区）
  0.696 ~ 1.0    = 0.304  ← 很疏（值稀疏区）
```

###### 用 NF4 量化 out0 行

先把 out0 行归一化到 [-1, 1]：

```
out0: [0.5, 0.1, 0.3, 6.0]

absmax = max(|0.5|, |0.1|, |0.3|, |6.0|) = 6.0

归一化：W_norm = W / absmax = [0.0833, 0.0167, 0.05, 1.0]
```

然后在 NF4 的 16 个值里找最近邻：

```
0.0833 → NF4 中找最近值：
  离 0.085（索引 9） 距离 = |0.0833-0.085| = 0.0017
  离 0.174（索引 10）距离 = |0.0833-0.174| = 0.0907
  离 0.008（索引 8） 距离 = |0.0833-0.008| = 0.0753
  → 选索引 9，值 0.085

0.0167 → 离 0.008（索引 8）距离 0.0087
         离 0.085（索引 9）距离 0.0683
         → 选索引 8，值 0.008

0.05 → 离 0.085（索引 9） 距离 0.035
        离 0.008（索引 8） 距离 0.042
        → 选索引 9，值 0.085

1.0 → 离 1.0（索引 15）距离 0 → 选索引 15，值 1.0
```

反量化还原（× absmax = 6.0）：

```
W_recon = [0.085×6.0, 0.008×6.0, 0.085×6.0, 1.0×6.0]
        = [0.51, 0.048, 0.51, 6.0]
```

**对比 INT4 和 NF4 的效果**：

```
原始值:   [0.5,  0.1,  0.3,  6.0]
INT4 量化: [0.857, 0,   0,   6.0] ❌ 小值全死
NF4 量化: [0.51, 0.048, 0.51, 6.0] ✅ 小值保住了
```

这就是 NF4 的优势——中间格子的密度大，小值的精度比 INT4 好得多。答主说"对正态分布数据比 INT4/FP4 更优"就是这个意思。

---

###### 二、第 2 步：冻结 W_int4，添加 LoRA

W_int4 现在冻结不动了。

```
W_int4 冻结 ← 不再更新
```

添加 LoRA：

```
对 out0 这一行（4 维输入 → 1 维输出），假设 rank=2：

LoRA_A: 4×2 矩阵（把 4 维降到 2 维）
LoRA_B: 2×1 矩阵（把 2 维升回 1 维）

原始 W 的参数量：4 × 4 = 16
LoRA 的参数量：(4×2 + 2×4) × 4行... 不对，按行来算。

实际 LoRA 是在整个层上加的：
对于 4×4 的 W（输出 4 维，输入 4 维）：
  W = W_int4_recon + LoRA_A × LoRA_B
  
  LoRA_A: 4×2  ← 维度 d×r
  LoRA_B: 2×4  ← 维度 r×d
  
  LoRA 参数量：4×2 + 2×4 = 16
  原始 W 参数量：4×4 = 16
  
  在这个小例子里 LoRA 和 W 一样大。但在大模型中：
  d=4096, r=8
  W=4096×4096=16M
  LoRA=4096×8+8×4096=65K
  LoRA 只有 W 的 0.4% ← 这才是 QLoRA 的典型场景
```

---

###### 三、第 3 步：前向传播

###### 量化部分

```
W_int4 → 反量化 → W_fp16_recon
W_fp16_recon_out0 = [0.51, 0.048, 0.51, 6.0]

Y_quant = X · W_fp16_reconᵀ

Y_quant[0][0] = token0 · out0_recon
  = 2.0×0.51 + 0.5×0.048 + 0.8×0.51 + 0.3×6.0
  = 1.02 + 0.024 + 0.408 + 1.8
  = 3.252

原始 FP16 Y[0][0] = 3.09
误差 = 3.252 - 3.09 = 0.162
```

###### LoRA 补偿部分

```
LoRA_A 随机初始化（假设值）：
LoRA_A_out = [0.02, -0.01, 0.03, 0.01]ᵀ  × 2 维

LoRA_B 初始化为 0：
LoRA_B = [0, 0]  ← 这样初始时 LoRA 输出为 0，不影响原始模型

初始时：Y_lora = 0
所以：Y_total = Y_quant + 0 = 3.252
```

###### 最终 Loss

```
Loss = MSE(Y_total, Y_true)  ← Y_true 是微调目标（某个下游任务的标签）
     = (3.252 - y_true)²
```

---

###### 四、第 4 步：反向传播 — 只更新 LoRA

```
Loss 反向传播：
  ∂Loss/∂W_int4 = 0  ← 不更新 W_int4（冻结）
  
  ∂Loss/∂LoRA_A = ... ← 更新 LoRA_A
  ∂Loss/∂LoRA_B = ... ← 更新 LoRA_B
  
LoRA_A_new = LoRA_A_old - lr × grad
LoRA_B_new = LoRA_B_old - lr × grad
```

经过多轮微调，LoRA 学会了补偿量化误差：

```
假设微调后 LoRA 学会了：
LoRA_A_out = [0.05, -0.02, 0.04, 0.01]
LoRA_B = [-1.0, 0.5]

LoRA 输出 = LoRA_A × LoRA_B
  = 0.05×(-1.0) + (-0.02)×0.5 + 0.04×(-1.0) + 0.01×0.5
  = -0.05 - 0.01 - 0.04 + 0.005
  = -0.095

Y_total = Y_quant + Y_lora
Y_total[0][0] = 3.252 + (-0.095)
  = 3.157

更接近原始 3.09 了！
```

---

###### 五、第 5 步：部署推理

微调完成后，部署时模型包含两部分：

```
部署的模型：
  1. W_int4[4×4] + scale 元数据 ← NF4 格式
  2. LoRA_A[4×2] + LoRA_B[2×4] ← FP16 格式

推理前向：
  W_int4 → 反量化 → W_fp16_recon
  Y = X · W_fp16_reconᵀ + X · (LoRA_A · LoRA_B)ᵀ
```

---

###### 六、答主还提到了两个关键技术

###### 1. 双量化（Double Quantization）

```
W 量化后，每组的 scale 也要存：
  假设 group=64，每 64 个权重共享一个 FP32 scale
  额外开销 = 32bit / 64个参数 = 0.5 bit/参数

双量化：把 scale 也量化，比如从 FP32 量化到 FP8
  额外开销 = 8bit / 64个参数 = 0.125 bit/参数
  节省了 0.375 bit/参数！
```

###### 2. 分页优化器（Paged Optimizers）

```
微调时，优化器状态（Adam 的动量、方差等）占大量显存。

QLoRA 的分页优化器：
  放不下的优化器状态 → 自动卸载到 CPU 内存
  需要时 → 加载回 GPU
  类似操作系统的虚拟内存/页面交换
```

---

###### 七、整体对比：QLoRA vs LLM-QAT vs GPTQ

用我们 out0 行的 [0.5, 0.1, 0.3, 6.0] 看三种方法的差异：

```
原始 out0: [0.5, 0.1, 0.3, 6.0]

GPTQ（PTQ）：
  W 被改为：[0.5, 0.15, 0.3, 6.0]  ← 动了原始值（补偿调整）
  量化后：  [0.857, 0, 0, 6.0]
  无额外参数
  只需要几分钟

LLM-QAT（QAT）：
  训练过程中 W 被微调为：[0.48, 0.12, 0.28, 5.9]
  量化后：               [0.479, 0.126, 0.283, 5.9]
  无额外参数
  需要全量训练

QLoRA（QAF）：
  W 不动：               [0.5, 0.1, 0.3, 6.0]
  量化后（NF4）：         [0.51, 0.048, 0.51, 6.0]
  额外 LoRA 参数补偿误差
  只需要少量微调数据
```

---

**一句话总结 QLoRA**：模型量化为 4bit 冻结不动，加一个很小的 LoRA 旁路去学习"如何补偿量化误差"。训练成本低，精度比 PTQ 好，比 QAT 省。

## 2.4 LLM.int8()

**特点**：W8A8、推理性能差、在模型加载时进行量化、不需要校准集

LLM.int8() 是一种采用**混合精度分解**的量化方法。该方案先做了一个矩阵分解，对绝大部分权重和激活用 8bit 量化（vector-wise）。对离群特征的几个维度保留 16bit，对其做高精度的矩阵乘法。

### 三个步骤

1. 从输入的隐含状态中，按列提取异常值（离群特征，即大于某个阈值的值）。
2. 对离群特征进行 FP16 矩阵运算，对非离群特征进行量化，做 INT8 矩阵运算。
3. 反量化非离群值的矩阵乘结果，并与离群值矩阵乘结果相加，获得最终的 FP16 结果。

### 用数据一步步推演

**第 1 步：检测离群值列（阈值 = 6.0）**

```
X 激活矩阵（2×4）：
         c0    c1    c2    c3
token0:  [2.0,  0.5,  0.8,  0.3]
token1:  [0.4,  7.0,  0.2,  0.6]

按列看最大值：
c0: max(2.0,0.4)=2.0 <6.0 → 正常列 ✅
c1: max(0.5,7.0)=7.0 ≥6.0 → 离群列 ❌
c2: max(0.8,0.2)=0.8 <6.0 → 正常列 ✅
c3: max(0.3,0.6)=0.6 <6.0 → 正常列 ✅

检测到 X 的 c1 是离群列 → W 的 c1 列也用 FP16 算
```

> ⚠️ **易错校正：历史推演曾把 c3 也误判为离群列**
>
> 早期学习推演中曾把 c3 也标为离群列（理由是"W 的 out0 第 4 个元素是 6.0"）。**这是笔误**：LLM.int8() 检测离群值针对的是**激活 X 的列**，而 X 的 c3 列 max(0.3, 0.6)=0.6，远小于阈值 6.0，是正常列。那个 6.0 是 **W 的 out0 行**第 4 个元素，不是激活列的最大值——离群值是激活里的现象（虽然它对应 W 的该列一起走 FP16）。正确的离群列判定只有 c1。

> 🔍 **关键点：检测的是 X 的列，对应 W 的也是列（不是行）**
>
> 因为 `Y[i][j] = Σₖ X[i][k] × W[j][k]`：X 的第 k 列 和 W 的所有行的第 k 列对应。所以离群值按 **X 的列** 检测，取出来的是 **W 的对应列**。

**第 2 步：拆分矩阵**

```
正常部分（INT8 计算，只保留 c0, c2, c3）：
  X_norm(2×3)、W_norm(4×3)

离群部分（FP16 计算，只保留 c1）：
  X_outlier(2×1)、W_outlier(4×1)
```

**第 3 步：对正常部分做 INT8 量化 + 矩阵乘**

```
对 X_norm 做 per-token INT8 量化：
  token0: [2.0,0.8,0.3] → s=2.0/127=0.0157 → [127, 51, 19]
  token1: [0.4,0.2,0.6] → s=0.6/127=0.0047 → [85, 43, 127]

对 W_norm 做 per-channel INT8 量化：
  out0: [0.5,0.3,6.0] → s=6.0/127=0.0472 → [11, 6, 127]
  out1: [0.4,0.6,0.8] → s=0.8/127=0.0063 → [63, 95, 127]
  ...

INT8 矩阵乘：
  Y_norm_int32[0][0] = 127×11 + 51×6 + 19×127 = 4116
  反量化：4116 × 0.0157 × 0.0472 = 3.049
```

**第 4 步：对离群部分做 FP16 矩阵乘**

```
Y_outlier_fp16[0][0] = 0.5 × 0.1 = 0.05
```

**第 5 步：合并结果**

```
Y_fp16 = Y_norm_fp16 + Y_outlier_fp16
Y[0][0] = 3.049 + 0.05 = 3.099 ✅（原始 3.09，误差 0.009）
```

### LLM.int8() 为什么不做纯 INT8？

LLM.int8() 论文发现：当模型规模超过 6.7B 参数时，**离群值会系统性出现**，集中在某些特征维度上，值可以非常大（几十到几百）。这时候如果不分离出来，INT8 会严重截断。

### LLM.int8() 的优缺点

**优点**：不需要校准集（加载时直接量化，开箱即用）；精度损失极小（离群值用 FP16 保住）；大部分计算是 INT8。

**缺点**（答主原文说"推理性能差"）：每一层都要做检测离群列→拆分矩阵→两个 GEMM→合并结果，额外开销固定存在；随着批量增大，离群值占比变大，劣势更明显。

> 🔍 **LLM.int8() 像消防员，SmoothQuant 像建筑师**
>
> - **LLM.int8()**：看到着火（离群值）了再去救——运行时检测、拆分、混合精度。
> - **SmoothQuant**：提前把房子设计成不容易着火的结构——离线平滑变换，推理时一个 INT8 GEMM 搞定。

> 🔍 **学习追问：LLM.int8() 是 W 和 A 都 INT8、W 离线整好、A 运行时当场检测离群值？**
>
> **基本完全正确**，只有一个小细节：
> | 你的理解 | 对不对 |
> |:------|:------|
> | 权重和激活都是用 INT8（W8A8） | ✅ |
> | W 离线整好 | ✅ 权重量化是离线做的 |
> | 激活运行时当场检测离群值 | ✅ 动态检测 |
> | 有离群值的列和对应的 W 单独做 FP16 | ✅ 但注意是 W 的**列**，不是行 |
> | 其他的做 INT8 计算 | ✅ |

> 🔍 **学习追问：阈值 6.0 是离线确定的吗？**
>
> **对，是离线确定的固定超参数。** LLM.int8() 论文中作者通过分析大量模型的激活值分布，经验性地把阈值定在 6.0，固定写死，不会在推理时动态调整（HuggingFace/bitsandbytes 中 `llm_int8_threshold` 默认也是 6.0，可调但属于超参数）。真正的 outlier activation 是运行时根据 hidden states 检测/处理的。
>
> | 什么 | 离线还是在线 |
> |:----|:-----------|
> | 权重 W 的 INT8 量化参数 (s,z) | **离线** |
> | 激活 X 的 INT8 量化参数 (s,z) | **在线**（运行时动态算） |
> | 离群值检测阈值（6.0） | **离线定好，推理时不改** |
> | 离群值的检测动作本身 | **在线**（每条数据进来都扫一遍） |

---

### 📌 原始 Q&A：LLM.int8()（历史第 42-46 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 42 轮）

LLM.int8()
这部分呢？

##### A（历史第 43 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！用我们的 4×4 数据推演 **LLM.int8()**。

---

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

激活 X(2×4)：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]
```

---

###### 一、LLM.int8() 的核心思想

答主原文：

> LLM.int8() 是一种采用**混合精度分解**的量化方法。该方案先做了一个矩阵分解，对绝大部分权重和激活用 8bit 量化（vector-wise）。对离群特征的几个维度保留 16bit，对其做高精度的矩阵乘法。

**一句话**：大部分用 INT8 算得快，少部分离群值用 FP16 保精度。**W8A8 混合精度**。

---

###### 二、三个步骤

答主原文给出的三步骤：

> 1. 从输入的隐含状态中，按列提取异常值（离群特征，即大于某个阈值的值）。
> 2. 对离群特征进行 FP16 矩阵运算，对非离群特征进行量化，做 INT8 矩阵运算。
> 3. 反量化非离群值的矩阵乘结果，并与离群值矩阵乘结果相加，获得最终的 FP16 结果。

---

###### 三、用数据一步步推演

###### 第 1 步：检测离群值列

```
X 激活矩阵（2×4）：
         c0    c1    c2    c3
token0:  [2.0,  0.5,  0.8,  0.3]
token1:  [0.4,  7.0,  0.2,  0.6]

按列看每个 token 的值，LLM.int8() 的阈值为 6.0：

c0: max(2.0, 0.4) = 2.0  < 6.0 → 正常列 ✅
c1: max(0.5, 7.0) = 7.0  ≥ 6.0 → **离群列** ❌
c2: max(0.8, 0.2) = 0.8  < 6.0 → 正常列 ✅
c3: max(0.3, 0.6) = 0.6  < 6.0 → 正常列 ✅
```

再看权重 W 和这个检测的关系。注意 LLM.int8() 检测的是 **激活的列**，但受影响的权重是 **权重的对应列**：

```
检测到 X 的 c1 是离群列 → W 的 c1 列也会用 FP16 算

W:
         c0    c1    c2    c3
out0:  [0.5,  0.1,  0.3,  6.0]  ← c1 列被标记
out1:  [0.4,  0.2,  0.6,  0.8]
out2:  [0.7,  0.3,  0.5,  0.2]
out3:  [0.1,  0.9,  0.4,  0.6]
```

---

###### 第 2 步：拆分矩阵

**正常部分（INT8 计算）—— 只保留 c0, c2, c3 列**：

```
X_norm（2×3）：
         c0    c2    c3
token0: [2.0,  0.8,  0.3]
token1: [0.4,  0.2,  0.6]

W_norm（4×3）—— 每行只保留 c0, c2, c3：
         c0    c2    c3
out0:  [0.5,  0.3,  6.0]
out1:  [0.4,  0.6,  0.8]
out2:  [0.7,  0.5,  0.2]
out3:  [0.1,  0.4,  0.6]
```

**离群部分（FP16 计算）—— 只保留 c1 列**：

```
X_outlier（2×1）：
         c1
token0: [0.5]
token1: [7.0]

W_outlier（4×1）—— 每行只保留 c1：
         c1
out0:  [0.1]
out1:  [0.2]
out2:  [0.3]
out3:  [0.9]
```

---

###### 第 3 步：对正常部分做 INT8 量化 + 矩阵乘

**对 X_norm 做 per-token INT8 量化**：

```
X_norm:
         c0    c2    c3
token0: [2.0,  0.8,  0.3]  → s_token0 = 2.0/127 = 0.0157
                              2.0→127, 0.8→51, 0.3→19
token1: [0.4,  0.2,  0.6]  → s_token1 = 0.6/127 = 0.0047
                              0.4→85, 0.2→43, 0.6→127

X_norm_int8:
         c0   c2   c3
token0: [127,  51,  19]
token1: [85,   43,  127]
```

**对 W_norm 做 per-channel INT8 量化**：

```
W_norm:
         c0    c2    c3
out0:  [0.5,  0.3,  6.0]  → max(0.5, 0.3, 6.0)=6.0, s=6.0/127=0.0472
                              0.5→11, 0.3→6, 6.0→127
out1:  [0.4,  0.6,  0.8]  → max=0.8, s=0.8/127=0.0063
                              0.4→63, 0.6→95, 0.8→127
out2:  [0.7,  0.5,  0.2]  → max=0.7, s=0.7/127=0.0055
                              0.7→127, 0.5→91, 0.2→36
out3:  [0.1,  0.4,  0.6]  → max=0.6, s=0.6/127=0.0047
                              0.1→21, 0.4→85, 0.6→127

W_norm_int8:
         c0   c2   c3
out0:  [ 11,   6,  127]
out1:  [ 63,  95,  127]
out2:  [127,  91,   36]
out3:  [ 21,  85,  127]
```

**做 INT8 矩阵乘**：

```
Y_norm_int32 = X_norm_int8(2×3) × W_norm_int8ᵀ(3×4)

Y_norm_int32[0][0] = token0_norm · out0_norm
  = 127×11 + 51×6 + 19×127
  = 1397 + 306 + 2413
  = 4116

Y_norm_int32[1][0] = token1_norm · out0_norm
  = 85×11 + 43×6 + 127×127
  = 935 + 258 + 16129
  = 17322
```

**反量化回 FP16**：

```
Y_norm_fp16[0][0] = 4116 × s_token0 × s_out0_norm
  = 4116 × 0.0157 × 0.0472
  = 4116 × 0.000741
  = 3.049

Y_norm_fp16[1][0] = 17322 × s_token1 × s_out0_norm
  = 17322 × 0.0047 × 0.0472
  = 17322 × 0.000222
  = 3.845
```

---

###### 第 4 步：对离群部分做 FP16 矩阵乘（不量化）

```
Y_outlier_fp16 = X_outlier(2×1) × W_outlierᵀ(1×4)

Y_outlier_fp16[0][0] = token0_c1 × out0_c1
  = 0.5 × 0.1 = 0.05

Y_outlier_fp16[1][0] = token1_c1 × out0_c1
  = 7.0 × 0.1 = 0.70
```

---

###### 第 5 步：合并结果

```
Y_fp16 = Y_norm_fp16 + Y_outlier_fp16

Y[0][0] = 3.049 + 0.05  = 3.099  ✅
Y[1][0] = 3.845 + 0.70  = 4.545

对照原始 FP16 结果：
Y_original[0][0] = token0 · out0
  = 2.0×0.5 + 0.5×0.1 + 0.8×0.3 + 0.3×6.0
  = 1.0 + 0.05 + 0.24 + 1.8
  = 3.09

误差 = 3.099 - 3.09 = 0.009  ← 很小 ✅

Y_original[1][0] = token1 · out0
  = 0.4×0.5 + 7.0×0.1 + 0.2×0.3 + 0.6×6.0
  = 0.2 + 0.7 + 0.06 + 3.6
  = 4.56

误差 = 4.545 - 4.56 = -0.015  ← 也很小 ✅
```

---

###### 四、如果不用混合精度（纯 INT8）会怎样？

对比一下：如果把 c1 列也 INT8 量化，不分离出去：

```
X_c1_token1 = 7.0
s = 7.0/127 = 0.0551
7.0 → 127 → 反量化 7.0 ✅（刚好在边界）

但 W_c1_out1 = 0.2
s = 0.9/127 = 0.0071（per-channel 按行算，out3 有 0.9 拉高了）
0.2 → 28 → 反量化 0.199 ← 还行

但是如果 X 的 c1 列出现更大的值，比如 12.0：
12.0 → 127×12.0/s... 不对，s=12.0/127=0.0945
12.0 → 127 → 反量化 12.0 ✅

好，看起来纯 INT8 也能处理？不对，问题在于 W 的 c1 列也有 out3 的 0.9 拉高了 s。
而且 LLM.int8() 的阈值 6.0 是经验值，对于更大的 LLM，离群值可以大到几十甚至上百。
```

**LLM.int8() 的发现**：当模型规模超过 6.7B 参数时，**离群值会系统性出现**，集中在某些特征维度上，而且值可以非常大（几十到几百）。这时候如果不分离出来，INT8 会严重截断。

---

###### 五、LLM.int8() 的优缺点

###### 优点

- **不需要校准集** — 加载时直接量化，开箱即用
- **精度损失极小** — 离群值用 FP16 保住了精度
- **W8A8** — 大部分计算是 INT8，快

###### 缺点

答主原文说：

> LLM.int8() 的**推理性能差**

为什么？因为每一层都要做：

```
1. 检测离群值列 ← 额外开销
2. 拆分成两个矩阵 ← 额外开销
3. INT8 GEMM (大部分) ← 快
4. FP16 GEMM (小部分) ← 慢
5. 合并两个结果 ← 额外开销
```

**随着批量增大，劣势更明显**：

```
batch=1 时：离群值占比可能 1%，大部分用 INT8，还行
batch=16 时：离群值占比可能到 5%-10%，FP16 部分变大
而且矩阵拆分/合并的开销固定存在
```

相比之下，SmoothQuant 不需要拆分矩阵，直接一个 INT8 GEMM 搞定，所以推理速度更快。

---

###### 六、对比 SmoothQuant

两个方法都做 W8A8，但思路完全不同：

```
LLM.int8():
  X → 检测离群列 → 拆成 [X_norm, X_outlier]
  W → 对应拆成 [W_norm, W_outlier]
  Y = INT8_GEMM(X_norm, W_norm) + FP16_GEMM(X_outlier, W_outlier)
  推理时做决定 ← 运行时开销

SmoothQuant:
  X' = X / s
  W' = W × s
  Y = INT8_GEMM(X', W')  ← 不拆分，一个 GEMM 搞定
  平滑因子离线算好 ← 运行时零开销
```

**LLM.int8() 像个消防员**：看到着火（离群值）了再去救。  
**SmoothQuant 像个建筑师**：提前把房子设计成不容易着火的结构。

---

###### 七、一张图总结 LLM.int8()

```
输入 X(2×4) + W(4×4)

         ↓
    检测离群列：c1 是离群列（7.0 ≥ 6.0）
         ↓
┌─────────────────────────────────────────┐
│  拆分矩阵                                │
│  ┌──────────┐  ┌──────────┐             │
│  │ 正常部分  │  │ 离群部分  │             │
│  │ c0,c2,c3  │  │ c1       │             │
│  ├──────────┤  ├──────────┤             │
│  │ INT8量化  │  │ FP16不量化│             │
│  │ INT8 GEMM│  │ FP16 GEMM│             │
│  │ 反量化    │  │          │             │
│  └──────────┘  └──────────┘             │
│         ↓              ↓                 │
│       Y_norm       Y_outlier             │
│         └──────┬──────┘                  │
│                ↓                         │
│          Y = Y_norm + Y_outlier          │
└─────────────────────────────────────────┘
         ↓
输出 Y(2×4)
```

需要继续推演 **GPTQ** 吗？

##### Q（历史第 44 轮）


---
再讨论下LLM.int8()这部分其实就是 无论是权重和激活都是用int8对吧，然后w的话就是离线整好；然后激活的话就是在运行的时候进行，然后当场根据数值检测 离群值，然后那个有离群值的 列就和对应的W的那个行或者列单独进行fp全精度的计算然后其它的还是int8计算？

##### A（历史第 45 轮，正文原样保留；仅将内部标题降级以保持目录层级）

你的理解 **基本完全正确**！只有一个小细节我帮你精确一下。

---

###### 你的理解 ✅

| 你说的 | 对不对 |
|:------|:------|
| 权重和激活都是用 INT8 | ✅ 对的，W8A8 |
| W 离线整好 | ✅ 权重量化是离线做的 |
| 激活运行时当场检测离群值 | ✅ 动态检测 |
| 有离群值的列和对应的 W 单独做 FP16 | ✅ 对的 |
| 其他的做 INT8 计算 | ✅ 对的 |

---

###### 一个小精确：检测的是 X 的**列**，对应 W 的也是**列**

不是 W 的行，是 **列**。因为矩阵乘法是：

```
Y = X · Wᵀ

Y[i][j] = Σₖ X[i][k] × W[j][k]
         ↑         ↑
     X 的第 k 列   W 的第 j 行的第 k 列
```

所以 X 的第 k 列 和 W 的所有行的第 k 列 对应。离群值是按 **X 的列** 检测的，取出来的是 **W 的对应列**。

在我们数据中：

```
X(2×4) 检测出 c1 是离群列（有 7.0）：
         c0    c1    c2    c3
token0: [2.0,  0.5,  0.8,  0.3]
token1: [0.4,  7.0,  0.2,  0.6]  ← c1 是离群列
                ↑

W(4×4) 取出 c1 列做 FP16（所有行的 c1 列）：
         c0    c1    c2    c3
out0:  [0.5,  0.1,  0.3,  6.0]
out1:  [0.4,  0.2,  0.6,  0.8]
out2:  [0.7,  0.3,  0.5,  0.2]
out3:  [0.1,  0.9,  0.4,  0.6]
              ↑
       c1 列全部取出做 FP16
```

---

###### 所以 LLM.int8() 的完整流程一句话

> 检测 X 的哪些**列**有离群值（值 > 6.0），把这些列和 W 对应的列一起用 FP16 算，其他列用 INT8 算，最后加起来。

**每条数据进来都要重新检测一次**，因为每次输入的 X 不同，离群列的位置可能不同。

---

要不要继续看下一个方法 **GPTQ** 了？

##### Q（历史第 46 轮）

那这个数值  6.0 也是离线确定的是吧？

##### A（历史第 47 轮，正文原样保留；仅将内部标题降级以保持目录层级）

对，是离线确定的。

LLM.int8() 论文中，作者通过分析大量模型的激活值分布，经验性地把阈值定在 **6.0**。这个值固定死了，不会在推理时动态调整。

不过严格来说，6.0 这个具体数字在不同实现里可能有差异。核心是：

> **阈值是离线定好的固定值，不是推理时根据当前输入动态算的。**

所以回到我们刚才说的话：

| 什么 | 离线还是在线 |
|:----|:-----------|
| 权重 W 的 INT8 量化参数 (s, z) | **离线** |
| 激活 X 的 INT8 量化参数 (s, z) | **在线**（运行时动态算） |
| **离群值检测阈值（如 6.0）** | **离线定好，推理时不改** |
| 离群值的检测动作本身 | **在线**（每条数据进来都扫一遍） |

需要继续看 **GPTQ** 吗？

## 2.5 GPTQ

**特点**：W4A16、量化速度慢、离线静态量化

GPTQ（论文：GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS）是一种仅权重量化方法，其中模型权重被量化为 int4 数值类型，而激活值则保留在 float16。在推理阶段，模型权重被动态地反量化回 float16 并在该数值类型下进行实际的运算。同 OBQ 一样，GPTQ 从单层量化的角度考虑，希望找到一个量化过的权重，使得新的权重和老的权重之间输出的结果差别最小。

GPTQ 将权重分组（如：128 列为一组）为多个子矩阵（block）。对某个 block 内的所有参数逐个量化，每个参数量化后，需要适当调整这个 block 内其他未量化的参数，以弥补量化造成的精度损失。因此，GPTQ 量化需要准备校准数据集。

### GPTQ 的核心思想

GPTQ 把量化看作一个**优化问题**：

```
量化一个权重后 → 调整同组内其他未量化的权重 → 补偿误差
```

**这不是四舍五入，而是有"补偿机制"的量化。** 同 OBQ（Optimal Brain Quantization）一样，GPTQ 从**单层量化的角度**考虑——希望找到一个量化过的权重，使得新的权重和老的权重之间输出的结果差别最小。其前身是 OBS（Optimal Brain Surgeon）和 OBQ。

### 用数据一步步推演（group=2，INT4）

**第 0 步：准备校准数据，算 Hessian 矩阵**

```
校准集一条样本：X_calib = [1.0, 0.5, 0.3, 0.2]
对于 out0 这一行，Hessian 矩阵 H = 2·X_calibᵀ·X_calib：
H = [1.0,  0.5,  0.3,  0.2]
    [0.5,  0.25, 0.15, 0.1]
    [0.3,  0.15, 0.09, 0.06]
    [0.2,  0.1,  0.06, 0.04]
对角线 = 每个权重的重要性；非对角线 = 权重之间的关联性
```

**第 1 步：分组**：`out0: [0.5, 0.1 | 0.3, 6.0]`（组A、组B）

**第 2 步：量化组A**：
- 0.5：s_A=0.5/7≈0.0714，0.5→7→0.5（误差0，精确）
- 0.1：0.1→round(1.4)=1→0.0714（误差0.0286），组A只剩2个元素，另一个已量化完，误差无法补偿 → 组A完成

**第 3 步：量化组B的第一个元素 0.3**（关键！）

```
组B: [0.3, 6.0]，s_B=6.0/7=0.8571
0.3 → round(0.3/0.8571)=0 → 反量化 0，误差 δ=0.3

调整组B内未量化的元素 6.0（GPTQ 的精髓）：
  调整量 = -(H_jj⁻¹) × H_ji × δ
         = -(1/0.04) × 0.06 × 0.3
         = -25 × 0.06 × 0.3 = -0.45
  6.0 → 6.0 + (-0.45) = 5.55   ← 6.0 被改成了 5.55！
```

**第 4 步：量化组B的第二个元素 5.55**：

```
5.55 → round(5.55/0.8571)=6 → 反量化 5.143，误差 0.407
没有剩余元素可调整 → 组B完成
```

**完整 out0 量化结果**：

```
out0 原始：   [0.5, 0.1, 0.3, 6.0]
out0 量化后： [0.5, 0.0714, 0, 5.143]
out0 被调整： 不    不      不   6.0→5.55→量化→5.143

对比 RTN 和 GPTQ：
  RTN:  [0.857, 0, 0, 6.0]      ← 0.5 和 0.3 都没了
  GPTQ: [0.5, 0.0714, 0, 5.143] ← 0.5 保住了！6.0 被调整了
```

**注意**：本例 group=2 时 GPTQ 误差比 RTN 大（因为每组只有 1 个元素可调整，补偿机制没发挥空间）。实际中 group=128：量化第 1 个元素后还有 127 个未量化可调整，越早量化的元素误差被越多人分担。

### Hessian 矩阵到底怎么来的？（详细推导）

**GPTQ 的目标**：量化权重后让层的输出变化最小。

```
原始输出：Y = X·w     量化后输出：Y_q = X·w_q
输出变化量：ΔY = X·(w-w_q)
用 MSE 衡量：Loss = ||ΔY||² = ||X·(w-w_q)||²
展开：Loss = (w-w_q)ᵀ · (Xᵀ·X) · (w-w_q)
令 δ=w-w_q，H=Xᵀ·X：Loss = δᵀ·H·δ
```

**H = Xᵀ·X 就是 Hessian 矩阵。** 用数据算：

```
X_calib = [1.0, 0.5, 0.3, 0.2]
H = X_calibᵀ · X_calib
H[i][j] = X_calib[i] × X_calib[j]（外积）

对角线 H[j][j] = X_calib[j]² —— 权重 j 对输出的影响程度（重要性）
非对角线 H[i][j] = X_calib[i]·X_calib[j] —— 权重 i,j 的关联性
```

**调整量公式**（量化 w_q 产生误差 δ_q 后，调剩余 w_j）：

```
调整量 = -(H_jj⁻¹) × H_jq × δ_q
```

用数据验证（量化 0.3 后调 6.0）：
- 0.3 对应 c2，6.0 对应 c3；H_sub = [[0.09,0.06],[0.06,0.04]]
- 调整 6.0：-(1/0.04)×0.06×0.3 = -0.45

> 🔍 **直观理解**：Hessian 对角线小（如 H₃₃=0.04）意味着该权重对输出影响小 → 扛的调整量大（"不重要"的权重多扛误差）；对角线大（如 H₁₁=0.25）意味着重要 → 调整量小（"重要"的权重少扛）。**GPTQ 的策略就是：让不重要的元素多扛误差，重要的元素少扛。**

### GPTQ 的三个创新点

1. **取消贪心算法**：OBS 采用贪心策略（先量化影响最小的参数）；GPTQ 发现直接按顺序量化对精度影响不大 → 参数矩阵每一行的量化可以做并行的矩阵计算（per-channel quantization），量化速度快了一个数量级。
2. **Lazy Batch-Updates**：OBQ 逐个更新权重，瓶颈在 GPU 内存带宽。GPTQ 延迟一部分参数的更新，一次处理多个（如 128）列，缓解带宽压力。
3. **Cholesky 分解**：用 Cholesky 分解求 Hessian 矩阵的逆，提前算好所有需要的信息，增强数值稳定性并减少计算量。

### GPTQ 的完整流程（伪代码）

```
对于每层：
  1. 准备校准数据，过一遍前向，收集激活值
  2. 算 Hessian 矩阵 H = 2·Xᵀ·X
  3. Cholesky 分解 H⁻¹ = LLᵀ
  4. 对权重矩阵，按 block 划分（如 128 列一个 block）
  5. 对每个 block：
     a. 量化当前列的所有权重（所有行一起做）
     b. 算量化误差
     c. 用 Cholesky 因子调整 block 内未量化的列
  6. 所有 block 处理完，量化完成
```

### GPTQ vs RTN vs SmoothQuant vs AWQ

用 out0 的数据 [0.5, 0.1, 0.3, 6.0] 对比：

```
原始值：     [0.5,  0.1,  0.3,  6.0]
RTN:         [0.857, 0,    0,    6.0]    ← 小值全死，没动原始值
SmoothQuant: [0.679, 0.1, 0.433, 8.742] ← 数学变换，不是微调
AWQ:         [0.548, 0.194, 0.212, 4.026] ← 数学变换，不是微调
GPTQ:        [0.5, 0.0714, 0, 5.143]     ← 真的改了 6.0→5.55→5.143
```

| 方法 | 是否动了 W 的原始值？ | 补偿机制 |
|:----|:-------------------|:--------|
| RTN | ❌ 没动 | 无 |
| SmoothQuant | ❌ 数学等价变换 | 无 |
| AWQ | ❌ 数学等价变换 | 无 |
| **GPTQ** | **✅ 动了** | **量化后调整其他权重来补偿误差** |

> 🔍 **学习追问 1：补偿的是"下一个"还是"组内剩余所有未量化的"？**
>
> **是组内剩余所有未量化的**，不是只传给紧挨着的下一个：
> ```
> 组内 [w1, w2, w3, w4]
> 量化 w1 → 误差 δ₁ → 调整 w2, w3, w4（三个都调）
> 量化 w2' → 误差 δ₂ → 调整 w3', w4'（两个都调）
> ...
> 每个误差都被分摊到所有剩下的元素上
> ```

> 🔍 **学习追问 2：是"累计误差传给下一个"吗？**
>
> **不是。** 每次量化的是**已经被调整过的值**，不是原始值，也不是"累积误差"穿下去：
> ```
> 量化 w1 → 有误差 → 调 w2,w3,w4（w2 变成 w2'）
> 量化 w2'（不是原始 w2）→ 有误差 → 调 w3',w4'
> ...
> ```

> 🔍 **学习追问 3：GPTQ 是跨行并行的，不是一行一行做**
>
> GPTQ 的关键加速：**量化第 k 列时，所有行的第 k 列一起量化，所有行的剩余列一起调整**。
> ```
>          c0     c1     c2     c3
> out0:  [0.5,   0.1,   0.3,   6.0]
> out1:  [0.4,   0.2,   0.6,   0.8]
> out2:  [0.7,   0.3,   0.5,   0.2]
> out3:  [0.1,   0.9,   0.4,   0.6]
> 量化 c0 列（所有行的 c0 一起量化）→ 算误差 → 调整 c1,c2,c3 列
> 量化 c1 列（已被调整过的值）→ 算误差 → 调整 c2,c3 列
> ...
> ```

> 🔍 **学习追问 4：本 group 的误差，剩下的 group 会补偿吗？**
>
> **不补偿。每个 group 是独立的，误差不会跨 group 传递。** 因为不同 group 的 scale 不同，不在同一个"量化空间"里：
> ```
> group A 的 scale s_A=0.5/7=0.0714（厘米尺）
> group B 的 scale s_B=6.0/7=0.8571（分米尺）
> group A 量完剩 0.0286cm 的误差，让 group B 调值补？
> 但 group B 的最小格子是 0.8571dm，根本补不了这么小的误差
> ```
> **那整体误差怎么办？** group=128 时，128 个元素里前 127 个的误差都可以被后面的补偿，只有最后一个元素的误差是残留的（占比仅 1/128）。而且 Hessian 对角线小的权重（重要性低）被留到最后量化，重要性高的在前面就量化好了，精度有保障。

> ⚠️ **易错校正：GPTQ 的两个"128" + 误差传播范围**
>
> 1. **quantization group_size=128**：决定哪些权重共享 scale/zero（现代 checkpoint 里）——**group 内 scale 独立，误差不跨 group**；
> 2. **algorithm block_size B=128**：Lazy Batch Update 一次处理多少列——**在当前 algorithm block 内逐列量化，当前 block 完成后把累计 error 一次性更新到所有后续未量化列**，即算法 block 内误差会传播到后续列。
>
> 两者可能都等于 128，但**概念完全不同**。GPTQ 原论文的 Lazy Batch Update 误差传播范围是算法 block 内（可跨量化 group），而 scale/zero 的量化分组是另一回事。

> 🔍 **GPTQ 面试一句话**
>
> GPTQ 是一种 layer-wise weight-only PTQ，通过 calibration activation 构造近似 Hessian，用 inverse-Hessian 信息在逐列量化时更新尚未量化权重，把当前 rounding error 补偿到后续权重；再通过 arbitrary order、lazy block update 和 Cholesky 把 OBQ 扩展到百亿/千亿参数模型。

---

**博主原文配图**（源自知乎原文）：

![原文图 6](https://picx.zhimg.com/v2-46de2ad7e2b45a2aeba549d0edd70fdb_r.jpg?source=2c26e567)


![原文图 7](https://pic1.zhimg.com/v2-6d44cb77a36df08e59e303b391fe37ad_r.jpg?source=2c26e567)


---

### 📌 原始 Q&A：GPTQ（历史第 48-58 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 48 轮）

我现在现在开始讨论 GPTQ
 这部分的内容

##### A（历史第 49 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！用我们的 4×4 数据完整推演 **GPTQ**。

---

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

激活 X(2×4)：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]
```

---

###### 一、GPTQ 的核心思想

答主原文：

> **GPTQ 还是从单层量化的角度考虑，希望找到一个量化过的权重，使的新的权重和老的权重之间输出的结果差别最小。**

**GPTQ 把量化看作一个优化问题**：

```
量化一个权重后 → 调整同组内其他未量化的权重 → 补偿误差
```

**这不是四舍五入，而是有"补偿机制"的量化。**

---

###### 二、GPTQ 的数学基础（OBQ 的改进）

GPTQ 的前身是 **OBS（Optimal Brain Surgeon）** 和 **OBQ（Optimal Brain Quantization）**。

核心思想：

```
对权重 w_q 做量化后，它对输出产生的误差是：
  δ = w_q - Q(w_q)  量化误差

然后调整同组内其他未量化的权重来补偿这个误差：
  w_others → w_others + 补偿量
```

补偿量由 **Hessian 矩阵** 决定——Hessian 告诉模型"哪些权重之间的关联性强，调整一个会影响另一个的输出。"

---

###### 三、用数据一步步推演

###### 第 0 步：准备校准数据，算 Hessian 矩阵

GPTQ 需要校准数据。假设校准集就一条样本：

```
X_calib = [1.0, 0.5, 0.3, 0.2]   ← 一条校准样本，4 维
```

对于 out0 这一行，Hessian 矩阵 H 是：

```
H = 2 · X_calibᵀ · X_calib

H = 2 × [1.0]  × [1.0, 0.5, 0.3, 0.2]
         [0.5]
         [0.3]
         [0.2]

H = 2 × [1.0,   0.5,   0.3,   0.2 ]
         [0.5,   0.25,  0.15,  0.1 ]
         [0.3,   0.15,  0.09,  0.06]
         [0.2,   0.1,   0.06,  0.04]

H 是一个 4×4 矩阵，对角线元素表示"每个权重的重要性"，
非对角线元素表示"权重之间的关联性"。
```

GPTQ 的关键优化：用 **Cholesky 分解** 提前算好 Hessian 的逆矩阵的 Cholesky 因子，这样后续更新时直接查，不用重复算。

---

###### 第 1 步：分组（Group-wise）

GPTQ 把权重分成组，比如 group=2：

```
out0: [0.5, 0.1 | 0.3, 6.0]
        组A         组B
```

每组内逐个量化，每量化一个权重，就调整组内其他未量化的权重。

---

###### 第 2 步：量化组A的第一个元素 0.5

**INT4 量化 0.5**：

```
INT4 范围 -8 ~ 7，对称量化，组A的 max=0.5

s_A = 0.5 / 7 ≈ 0.0714

0.5 → round(0.5/0.0714) = round(7.0) = 7 → 反量化 7×0.0714 = 0.5
  误差 δ = 0.5 - 0.5 = 0  ✅ 精确量化！
```

0.5 本身刚好是 7 个格子，没有误差 → 不需要调整组内其他元素。

---

###### 第 3 步：量化组A的第二个元素 0.1

**INT4 量化 0.1**：

```
s_A = 0.5/7 = 0.0714（同一个组，共用 scale）

0.1 → round(0.1/0.0714) = round(1.4) = 1 → 反量化 1×0.0714 = 0.0714
  误差 δ = 0.1 - 0.0714 = 0.0286
```

**有误差了！** 现在要调整组A内其他**未量化**的元素——但组A只有两个元素，另一个 0.5 已经量化完了，没有未量化的可以调整了。所以组A量化完成。

```
组A量化结果：[0.5, 0.1] → [0.5, 0.0714]

误差 = 0.0286，无法补偿
```

---

###### 第 4 步：量化组B的第一个元素 0.3

**INT4 量化 0.3**：

```
组B: [0.3, 6.0]
s_B = 6.0/7 = 0.8571

0.3 → round(0.3/0.8571) = round(0.35) = 0 → 反量化 0
  误差 δ = 0.3 - 0 = 0.3
```

**大误差！** 现在要调整组B内未量化的元素 **6.0**。

---

###### 第 5 步：调整 6.0 来补偿 0.3 的误差 ← GPTQ 的精髓

调整量由 Hessian 矩阵决定。

对于组B，我们只关心 0.3 和 6.0 这两个位置。从 Hessian 矩阵中提取对应的子矩阵：

```
Hessian 中 0.3 对应 c2，6.0 对应 c3：
H_sub = [H[c2][c2], H[c2][c3]]  = [0.09, 0.06]
        [H[c3][c2], H[c3][c3]]    [0.06, 0.04]

要调整的是 6.0（c3）：
调整量 = -(H[c3][c3]⁻¹) × H[c3][c2] × δ
       = -(1/0.04) × 0.06 × 0.3
       = -25 × 0.06 × 0.3
       = -0.45
```

简化来说：**Hessian 告诉你 0.3 和 6.0 的关联性，以及 6.0 的"重要性"**。因为 6.0 的 Hessian 对角线值 0.04 很小，说明它对输出的影响相对小，可以承受调整。

```
6.0 → 6.0 + (-0.45) = 5.55
```

**关键：6.0 被改成了 5.55！** 这就是 GPTQ 和 AWQ/SmoothQuant 的本质区别——**GPTQ 改了 W 的原始值**。

---

###### 第 6 步：量化组B的第二个元素 5.55（调整后的）

```
s_B = 6.0/7 = 0.8571（scale 由量化前的组内 max 决定，还是 6.0/7）

5.55 → round(5.55/0.8571) = round(6.48) = 6 → 反量化 6×0.8571 = 5.143
  误差 δ = 5.55 - 5.143 = 0.407
```

**还有误差，但没有未量化的元素可以调整了。** 组B量化完成。

```
组B量化结果：[0.3, 6.0] → [0, 5.143]
```

---

###### 第 7 步：完整 out0 量化结果

```
out0 原始：     [0.5, 0.1, 0.3, 6.0]
out0 量化后：   [0.5, 0.0714, 0, 5.143]
out0 被调整过：  不  不      不  从 6.0→5.55→量化→5.143
```

**对比 RTN 和 GPTQ 的 out0 结果**：

```
RTN:   [0.857, 0, 0, 6.0]     ← 0.5 和 0.3 都没了
GPTQ:  [0.5, 0.0714, 0, 5.143] ← 0.5 保住了！6.0 被调整了
```

---

###### 第 8 步：算 Y 对比精度

```
GPTQ 量化后的 Y：
Y[0][0] = token0 · out0_gptq
  = 2.0×0.5 + 0.5×0.0714 + 0.8×0 + 0.3×5.143
  = 1.0 + 0.0357 + 0 + 1.543
  = 2.579

原始 Y[0][0] = 3.09
GPTQ 误差 = 2.579 - 3.09 = -0.511 ❌

RTN 的 Y[0][0] = 3.514
RTN 误差 = 3.514 - 3.09 = 0.424 ❌
```

**GPTQ 的误差比 RTN 还大？** 这是因为我们的例子太小了（group=2，每个组只有一个元素可以被调整），GPTQ 的补偿机制没有发挥空间。

**在实际中（group=128）**：

```
GPTQ 的典型配置：
  group=128 个元素
  量化第 1 个元素后 → 还有 127 个未量化的可以调整
  量化第 2 个元素后 → 还有 126 个未量化的可以调整
  ...

越早量化的元素，它的误差可以被更多未量化的元素分担补偿。
而且 GPTQ 用的 Lazy Batch-Updates 一次处理 128 列，用矩阵运算整体优化。
```

---

###### 四、GPTQ 的三个创新点（答主原文）

###### ① 取消贪心算法

**OBS 的做法**：先量化对输出影响最小的参数（贪心，要排序）
**GPTQ 的做法**：直接按顺序量化，不影响精度

```
OBS: 找影响最小的 → 量化 → 找下一个影响最小的 → 量化 → ...
GPTQ: 按顺序量化第一个 → 调整剩下的 → 按顺序量化第二个 → ...

GPTQ 不需要"找"，所以每行可以做并行计算！
```

###### ② Lazy Batch-Updates

**不逐列更新，一次处理 128 列**。

```
一次性处理整个 block（128 列）：
  1. 算这 128 个权重的量化值
  2. 算这 128 个量化误差
  3. 一次性调整 block 内其余未量化的权重

减少 GPU 内存带宽压力，大幅提升速度。
```

###### ③ Cholesky 分解

提前算好 Hessian 矩阵逆的 Cholesky 因子，后续更新时直接查表：

```
H⁻¹ = LLᵀ  (L 是下三角矩阵)

更新时：
  调整量 = -L × Lᵀ × H_子矩阵 × 误差
  
不用每次重新算，数值稳定性更好。
```

---

###### 五、GPTQ 的完整流程（答主原文配图）

伪代码：

```
对于每层：
  1. 准备校准数据，过一遍前向，收集激活值
  2. 算 Hessian 矩阵 H = 2·Xᵀ·X
  3. Cholesky 分解 H⁻¹ = LLᵀ
  4. 对权重矩阵，按 block 划分（如 128 列一个 block）
  5. 对每个 block：
     a. 量化当前列的所有权重（所有行一起做）
     b. 算量化误差
     c. 用 Cholesky 因子调整 block 内未量化的列
  6. 所有 block 处理完，量化完成
```

---

###### 六、GPTQ vs RTN vs SmoothQuant vs AWQ

用我们 out0 的数据 [0.5, 0.1, 0.3, 6.0] 对比：

```
原始值：        [0.5,  0.1,  0.3,  6.0]

RTN:            [0.857, 0,    0,    6.0]    ← 小值全死，没动原始值
SmoothQuant:    [0.679, 0.1, 0.433, 8.742] ← 数学变换，不是微调
AWQ:            [0.548, 0.194, 0.212, 4.026] ← 数学变换，不是微调
GPTQ:           [0.5, 0.0714, 0, 5.143]     ← 真的改了 6.0→5.55→5.143
```

**关键区别**：

| 方法 | 是否动了 W 的原始值？ | 补偿机制 |
|:----|:-------------------|:--------|
| RTN | ❌ 没动 | 无 |
| SmoothQuant | ❌ 数学等价变换 | 无 |
| AWQ | ❌ 数学等价变换 | 无 |
| **GPTQ** | **✅ 动了** | **量化后调整其他权重来补偿误差** |

需要继续看 **SmoothQuant** 吗？

##### Q（历史第 50 轮）

所以GPTQ就是离线量化权重W的算法，并且是per-group的。然后就是 每组一个一个的。然后并且累计误差，然后对于下一个进行补偿。而且是直接改原始数值进行补偿的对吗？

##### A（历史第 51 轮，正文原样保留；仅将内部标题降级以保持目录层级）

**基本正确，有 3 个细微之处我帮你精确一下：**

---

###### 你的理解 ✅

| 你说的 | 对不对 |
|:------|:------|
| 离线量化权重 W | ✅ 对的 |
| 只量化权重，不动激活（W4A16） | ✅ 对的 |
| per-group | ✅ 对的 |
| 一组内一个一个来 | ✅ 对的 |
| 有补偿机制 | ✅ 对的 |
| 直接改原始数值 | ✅ 对的 |

---

###### 三个精确点

###### ① 补偿的不是"下一个"，而是**组内剩余所有未量化的**

不是只补偿紧挨着的下一个，而是：

```
组内 [w1, w2, w3, w4]

量化 w1 → 误差 δ₁ → 调整 w2, w3, w4（三个都调）
量化 w2 → 误差 δ₂ → 调整 w3, w4（两个都调）
量化 w3 → 误差 δ₃ → 调整 w4（一个调）
量化 w4 → 没有可调的了
```

每个误差都被 **分摊到所有剩下的元素** 上，不是只传给下一个。

---

###### ② 不是"累计误差"传给下一个，而是**每次调完后再量化**

```
量化 w1 → 有误差 → 调 w2,w3,w4
          w2 被调成了 w2'
量化 w2'（不是原来的 w2）→ 有误差 → 调 w3,w4
          w3 被调成了 w3'
量化 w3'（不是原来的 w3）→ 有误差 → 调 w4
          w4 被调成了 w4'
量化 w4' → 结束
```

每次量化的是**已经被调整过的值**，不是原始值，也不是"累积误差"穿下去。

---

###### ③ GPTQ 是**跨行并行**的，不是一行一行做

这是 GPTQ 的关键加速：**量化第 k 列时，所有行的第 k 列一起量化，所有行的剩余列一起调整。**

```
W 矩阵视角（不是行视角）：

         c0     c1     c2     c3
out0:  [0.5,   0.1,   0.3,   6.0]
out1:  [0.4,   0.2,   0.6,   0.8]
out2:  [0.7,   0.3,   0.5,   0.2]
out3:  [0.1,   0.9,   0.4,   0.6]

量化 c0 列（所有行的 c0 一起量化）：
  0.5, 0.4, 0.7, 0.1  → INT4
  算误差 → 调整 c1,c2,c3 列（所有行）

量化 c1 列（已经被调整过的值）：
  再量化 → 算误差 → 调整 c2,c3 列

量化 c2 列（已经被调整过两次的值）：
  再量化 → 算误差 → 调整 c3 列

量化 c3 列（已经被调整过三次的值）：
  再量化 → 结束
```

这就是答主原文说的：

> 这项改进使得参数矩阵每一行的量化可以做并行的矩阵计算（即 per-channel quantization）。对于大模型场景，这项改进使得量化速度快了一个数量级。

---

###### 一句话精确版

> **GPTQ 是离线 W4 per-group PTQ。一组内按列逐个量化（所有行并行），每次量化后，用 Hessian 信息调整组内剩余所有未量化的列（所有行并行），补偿量化误差。调整的是 W 的原始浮点值。**

需要继续看 **SmoothQuant** 吗？

##### Q（历史第 52 轮）

你确定一下，我想问个事情，就是 如果本group的整的还有误差，那么剩下的group会进行补偿吗？

##### A（历史第 53 轮，正文原样保留；仅将内部标题降级以保持目录层级）

**不补偿。每个 group 是独立的，误差不会跨 group 传递。**

原因很简单——**每个 group 有自己的 scale**。

---

###### 为什么不能跨 group？

GPTQ 的补偿机制依赖 **Hessian 矩阵**，而 Hessian 矩阵反映的是"组内元素之间的关联性"。

```
GPTQ 是按组划分的：

group A [0.5, 0.1]  ← scale s_A
group B [0.3, 6.0]  ← scale s_B（独立！）
```

**每个 group 各自为政：**

```
group A 量化过程：
  量化 0.5 → 调 0.1
  量化 0.1（已被调过的）→ 结束
  group A 剩余的误差：留在 group A 内部

group B 量化过程：
  从头开始，group B 的值是原始的 [0.3, 6.0]
  不会看到 group A 的误差
```

---

###### 为什么设计成这样？

因为不同 group 的 **scale 不同**，它们不在同一个"量化空间"里：

```
group A 的 scale s_A = 0.5/7 = 0.0714
group B 的 scale s_B = 6.0/7 = 0.8571

group A 的误差是按 s_A 的格子算的
group B 的补偿是按 s_B 的格子算的
两个空间的"误差单位"不同，没法跨组补偿
```

打个比方：

```
group A 是用厘米尺量的（s_A=0.0714）
group B 是用分米尺量的（s_B=0.8571）

group A 量完还剩 0.0286 cm 的误差
你说让 group B 调整值来补这 0.0286 cm？
但 group B 的最小格子是 0.8571 dm，根本补不了这么小的误差
```

---

###### 那整体的误差怎么办？

一个 group 内的残留误差，**就留在那了**，不会被后续补偿。

但这不意味着 GPTQ 的整体精度差，因为：

```
1. GPTQ 的 group size 是 128，不是我们例子里的 2
   128 个元素里，前 127 个的误差都可以被后面的补偿
   只有最后一个元素的误差是残留的 → 占比仅 1/128

2. 这个残留误差被 Hessian 矩阵加权了
   Hessian 对角线小的权重（重要性低）才被留到最后量化
   重要性高的权重在前面就量化好了，精度有保障
```

---

###### 一句话

> **每个 group 的补偿是封闭的，误差不跨组。GPTQ 通过把 group 设大（128），让"无补偿的最后一个元素"占比极小，来保证整体精度。**

##### Q（历史第 54 轮）

部分

This may or may not be related to the current task.</ide_selection>

---
我现在想问一个问题就是 用一group的2个数据给我详细举个例子 int4的

##### A（历史第 55 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！就用我们 out0 行的组 B **`[0.3, 6.0]`**，group=2，INT4，完整走一遍 GPTQ 的量化过程。

---

###### 设定

```
组 B: [0.3, 6.0]
INT4 对称量化：范围 -8 ~ 7

校准数据（算 Hessian 用）：
X_calib = [1.0, 0.5, 0.3, 0.2]  ← 一条校准样本

Hessian 矩阵中组 B 对应的子矩阵：
H_sub = [0.09, 0.06]    ← 对应 0.3 和 6.0 的位置
        [0.06, 0.04]
```

---

###### 第 1 步：确定组 B 的 scale

INT4 对称量化，范围 -8 ~ 7，z=0：

```
组 B: [0.3, 6.0]
max(|0.3|, |6.0|) = 6.0

s = 6.0 / 7 ≈ 0.8571
```

---

###### 第 2 步：量化第一个元素 0.3

```
0.3 → round(0.3 / 0.8571) = round(0.35) = 0

反量化：0 × 0.8571 = 0

量化误差 δ₁ = 0.3 - 0 = 0.3
```

**量化前原始值：0.3**
**量化后反推值：0**
**误差：+0.3**

---

###### 第 3 步：补偿 — 调整组内剩余元素 6.0

现在用 Hessian 信息来调整组内还没量化的 6.0。

调整量公式（简化版）：

```
对于要调整的元素 w_j（这里是 6.0）：
  调整量 = -(H_jj⁻¹) × H_ji × δ₁
  其中：
    H_jj = 6.0 对应的 Hessian 对角线值 = 0.04
    H_ji = 6.0 和 0.3 的 Hessian 交叉值 = 0.06
    δ₁ = 0.3 的量化误差 = 0.3

调整量 = -(1/0.04) × 0.06 × 0.3
       = -25 × 0.06 × 0.3
       = -0.45
```

**6.0 被调整为：6.0 + (-0.45) = 5.55**

---

###### 第 4 步：量化第二个元素 5.55（被调整后的值）

```
5.55 → round(5.55 / 0.8571) = round(6.48) = 6

反量化：6 × 0.8571 = 5.143

量化误差 δ₂ = 5.55 - 5.143 = 0.407
```

**量化前被调整的值：5.55（不是原始 6.0）**
**量化后反推值：5.143**
**误差：+0.407**

---

###### 第 5 步：组内没有剩余元素了，结束

组 B 量化完成：

```
原始值：    [0.3,  6.0]
量化后值：  [0,   5.143]
被调整值：   不    6.0 → 5.55 → 量化 → 5.143
```

**误差：0.3 的误差 0.3 没有被完全补偿，因为只有一个元素可调，而且调完后 5.55 量化还有 0.407 的误差。**

---

###### 如果 group 更大呢？

假设 group=4，比如 [0.3, 6.0, 0.5, 0.2]：

```
量化 0.3 → 误差 0.3 → 调 6.0, 0.5, 0.2（三个一起调）
量化 6.0'（已被调过的）→ 误差 → 调 0.5', 0.2'（两个一起调）
量化 0.5''（已被调过两次的）→ 误差 → 调 0.2''（一个调）
量化 0.2'''（已被调过三次的）→ 误差 → 结束
```

**越早被量化的元素，它的误差可以被更多剩余元素分担补偿。越晚量化的元素，误差越难被补偿。**

这就是为什么 GPTQ 的 group size 选 128 而不是 2——**128 个元素里，前 127 个的误差都有"后备"可调，只有最后一个误差是残留的。**

##### Q（历史第 56 轮）

那用4个元素的例子给我讲述下

##### A（历史第 57 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！设定一个 group=4 的例子，完整走一遍 GPTQ 的量化 + 补偿链条。

---

###### 设定

```
一个 group 4 个元素：
w = [0.3, 6.0, 0.5, 0.2]

INT4 对称量化，范围 -8 ~ 7
```

---

###### 第 0 步：确定组 scale

```
max(|0.3|, |6.0|, |0.5|, |0.2|) = 6.0

s = 6.0 / 7 ≈ 0.8571
```

整个组共用这一个 scale，从头到尾不变。

---

###### 第 1 步：量化 w₀=0.3

```
0.3 → round(0.3/0.8571) = round(0.35) = 0 → 反量化 0

误差 δ₀ = 0.3 - 0 = 0.3
```

**有误差了 → 调剩下的三个 [6.0, 0.5, 0.2]**

假设 Hessian 给的调整量分配：

```
调整 w₁(6.0): 6.0 + (-0.45) = 5.55
调整 w₂(0.5): 0.5 + (-0.08) = 0.42
调整 w₃(0.2): 0.2 + (-0.03) = 0.17

此时剩余未量化元素已被调整为：[5.55, 0.42, 0.17]
```

---

###### 第 2 步：量化 w₁=5.55（已被调整过的）

```
5.55 → round(5.55/0.8571) = round(6.48) = 6 → 反量化 5.143

误差 δ₁ = 5.55 - 5.143 = 0.407
```

**又有误差了 → 调剩下的两个 [0.42, 0.17]**

```
调整 w₂(0.42): 0.42 + (-0.15) = 0.27
调整 w₃(0.17): 0.17 + (-0.06) = 0.11

此时剩余未量化元素已被调整为：[0.27, 0.11]
```

---

###### 第 3 步：量化 w₂=0.27（已被调整过两次的）

```
0.27 → round(0.27/0.8571) = round(0.31) = 0 → 反量化 0

误差 δ₂ = 0.27 - 0 = 0.27
```

**调剩下的最后一个 [0.11]**

```
调整 w₃(0.11): 0.11 + (-0.04) = 0.07

此时剩余未量化元素：[0.07]
```

---

###### 第 4 步：量化 w₃=0.07（已被调整过三次的）

```
0.07 → round(0.07/0.8571) = round(0.08) = 0 → 反量化 0

误差 δ₃ = 0.07 - 0 = 0.07
```

**没有剩余元素了，这个误差 0.07 残留下来，无法补偿。**

---

###### 完整流程回顾

```
原始值：     [0.3,       6.0,       0.5,       0.2      ]
               ↓           ↓           ↓           ↓
第1步 量化 0.3 → 0     调 6.0→5.55  调 0.5→0.42  调 0.2→0.17
               ↓                       ↓           ↓
第2步 量化 5.55 → 6                调 0.42→0.27  调 0.17→0.11
               ↓                                   ↓
第3步 量化 0.27 → 0                             调 0.11→0.07
               ↓
第4步 量化 0.07 → 0

最终量化值： [0,         5.143,     0,          0       ]
原始值：     [0.3,       6.0,       0.5,       0.2      ]
是否被调过：  没被调   调了1次    调了2次     调了3次
                         6.0→5.55  0.5→0.42   0.2→0.17
                                              0.42→0.27  0.17→0.11
                                                         0.11→0.07
```

---

###### 关键观察

| 元素 | 原始值 | 最终量化值 | 被调次数 | 是否能补偿自己的误差？ |
|:---:|:-----:|:---------:|:--------:|:-------------------:|
| w₀=0.3 | 0.3 | 0 | 0 | 误差 0.3 被 w₁,w₂,w₃ 分担了 |
| w₁=6.0 | 6.0 | 5.143 | 1 | 误差 0.407 被 w₂,w₃ 分担了 |
| w₂=0.5 | 0.5 | 0 | 2 | 误差 0.27 被 w₃ 分担了 |
| w₃=0.2 | 0.2 | 0 | 3 | **误差 0.07 没人可调，残留** |

**越早量化的元素，误差被补偿得越充分。最后一个元素是"冤大头"，所有误差都在它这终结。**

---

###### 对比 group=2 和 group=4

```
group=2 时：
  [0.3, 6.0] → [0, 5.143]
  0.3 的误差被 6.0 分担了，但 6.0 自己的误差残留

group=4 时：
  [0.3, 6.0, 0.5, 0.2] → [0, 5.143, 0, 0]
  0.3 的误差被 6.0,0.5,0.2 三个分担 → 0.3 的补偿更充分
  6.0 的误差被 0.5,0.2 两个分担
  0.5 的误差被 0.2 一个分担
  0.2 的误差残留
```

**group 越大，能被分摊的"后备"越多，整体精度越好。** 这就是为什么 GPTQ 用 group=128。

##### Q（历史第 58 轮）

好的我理解了这部分的调整。但是这个矩阵到底是怎么来的呢？我不是很清楚。是有很复杂的公式吗？

##### A（历史第 59 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好问题！这个矩阵其实不复杂，我直接从 GPTQ 的目标出发，一步步推导给你看。

---

###### GPTQ 在干什么？

GPTQ 的目标是：

> 量化权重后，让**层的输出 Y 的变化最小**

对于单行权重 w（比如 out0 行），数学上写：

```
原始输出：    Y = X · w                  ← X 是校准数据
量化后输出：  Y_q = X · w_q

输出变化量：  ΔY = Y - Y_q = X · (w - w_q)
```

GPTQ 想让 **ΔY 尽可能小**。

---

###### 从目标到矩阵

ΔY 的大小用 MSE 衡量：

```
Loss = ||ΔY||² = ||X · (w - w_q)||²
```

展开：

```
Loss = (w - w_q)ᵀ · (Xᵀ·X) · (w - w_q)

令 δ = w - w_q（量化误差向量）
令 H = Xᵀ·X

Loss = δᵀ · H · δ
```

**这个 H = Xᵀ·X 就是 Hessian 矩阵。**

---

###### 用我们的数据算一遍

校准数据 X_calib = [1.0, 0.5, 0.3, 0.2]

```
H = X_calibᵀ · X_calib

H = [1.0] × [1.0, 0.5, 0.3, 0.2]
    [0.5]
    [0.3]
    [0.2]

H = [1.0×1.0, 1.0×0.5, 1.0×0.3, 1.0×0.2]
    [0.5×1.0, 0.5×0.5, 0.5×0.3, 0.5×0.2]
    [0.3×1.0, 0.3×0.5, 0.3×0.3, 0.3×0.2]
    [0.2×1.0, 0.2×0.5, 0.2×0.3, 0.2×0.2]

H = [1.0,  0.5,  0.3,  0.2]
    [0.5,  0.25, 0.15, 0.1]
    [0.3,  0.15, 0.09, 0.06]
    [0.2,  0.1,  0.06, 0.04]

（实际 GPTQ 中 H = 2·Xᵀ·X，但 2 倍不影响比例关系，省略）
```

---

###### Hessian 矩阵的每项代表什么

```
H = [1.0,  0.5,  0.3,  0.2]      ← c0 列
    [0.5,  0.25, 0.15, 0.1]      ← c1 列
    [0.3,  0.15, 0.09, 0.06]     ← c2 列
    [0.2,  0.1,  0.06, 0.04]     ← c3 列
```

**对角线 —— 权重的重要性**

```
H[0][0] = 1.0  ← w₀（对应 0.3）非常重要，动它对输出影响大
H[1][1] = 0.25 ← w₁（对应 6.0）比较重要
H[2][2] = 0.09 ← w₂（对应 0.5）不太重要
H[3][3] = 0.04 ← w₃（对应 0.2）最不重要
```

**非对角线 —— 权重之间的关联性**

```
H[0][1] = 0.5  ← w₀ 和 w₁ 关联性强
H[1][3] = 0.1  ← w₁ 和 w₃ 关联性中等
H[2][3] = 0.06 ← w₂ 和 w₃ 关联性弱
```

---

###### 那调整量是怎么算的？

当我量化了 w₀（对应 0.3），误差 δ₀=0.3，我要调 w₁,w₂,w₃ 来补偿。

对每个剩余元素 wⱼ，调整量公式是：

```
调整量 = -(H_jj⁻¹) × H_jq × δ_q

其中：
  H_jj = wⱼ 自己的对角线（"你扛得住调吗？"）
  H_jq = wⱼ 和 w_q 的交叉项（"你和被量化的那个关联大吗？"）
  δ_q  = 量化误差（"要补多少？"）
```

###### 调 w₁=6.0：

```
H₁₁ = 0.25    ← w₁ 的对角线
H₁₀ = 0.5     ← w₁ 和 w₀ 的交叉项
δ₀  = 0.3

调整量 = -(1/0.25) × 0.5 × 0.3  = -4 × 0.5 × 0.3 = -0.6

6.0 → 6.0 + (-0.6) = 5.4
```

###### 调 w₂=0.5：

```
H₂₂ = 0.09
H₂₀ = 0.3
δ₀  = 0.3

调整量 = -(1/0.09) × 0.3 × 0.3  = -11.1 × 0.3 × 0.3 = -1.0

0.5 → 0.5 + (-1.0) = -0.5
```

###### 调 w₃=0.2：

```
H₃₃ = 0.04
H₃₀ = 0.2
δ₀  = 0.3

调整量 = -(1/0.04) × 0.2 × 0.3  = -25 × 0.2 × 0.3 = -1.5

0.2 → 0.2 + (-1.5) = -1.3
```

---

###### 直观理解

看对角线 H₃₃=0.04 最小，意味着 **w₃(0.2) 对输出的影响最小**，所以它扛的调整量最大（-1.5，直接从 0.2 被调成了 -1.3）。

相反，H₁₁=0.25 大，意味着 **w₁(6.0) 重要**，调整量小（-0.6，从 6.0 调到 5.4）。

**GPTQ 的策略**：让"不重要"的元素多扛误差，"重要"的元素少扛。

---

###### 一句话总结 Hessian 矩阵怎么来的

> **H = X_calibᵀ · X_calib，其中 X_calib 是校准数据。对角线表示"这个权重有多重要"，非对角线表示"这两个权重的关联有多强"。调整量公式就是"用不重要的权重去吸收重要权重的量化误差"。**

需要我用上面的 4 个元素例子，把完整 Hessian 算一遍，然后走完整个 GPTQ 流程吗？

## 2.6 SmoothQuant

**特点**：离线静态量化、逐通道对权重缩放，逐 Token 对激活缩放，W8A8（对权重进行 per-tensor 或 per-channel，对激活进行 per-token 或 per-tensor）

SmoothQuant 是一种同时确保准确率且推理高效的训练后量化 (PTQ) 方法，可实现 8 比特权重与激活 (W8A8) 量化。

针对大模型而言，权重很容易量化，对于激活值，由于异常值的存在，导致激活则较难量化。作者对比了 per-channel、per-token、per-tensor 激活量化方案。per-tensor 量化是最高效的实现方式，但只有逐通道量化（per-channel）保留了精度，而它与 INT8 GEMM Kernel 不兼容（即 per-channel 量化不能很好地映射到硬件加速的 GEMM 内核，硬件不能高效执行，从而增加了计算时间）。

为了进行 vector-wise quantization 以有效利用 INT8 GEMM Kernel，只能使用外部维度（即激活的维度 T 和权重的维度 C₀）的缩放因子，不能使用内部维度（即通道内维度 Cᵢ）。因此，先前的工作对激活都采用了 per-token 量化，但并不能降低激活的难度。

于是 SmoothQuant 提出了一种**数学上等价的逐通道缩放变换**（per-channel scaling transformation），通过引入平滑因子 s 来平滑激活异常值，数学上等效的变换将量化难度从激活转移到权重上，从而使模型易于量化，保持精度的同时，还能够保证提升推理速度。

根据量化方式不同，作者提出三种策略 O1、O2、O3，其计算延迟依次降低。SmoothQuant 的 O1 和 O2 级成功地保持了浮点精度，而 O3 级（per-tensor static）虽然延迟更低，但平均精度下降了 0.8%，可能是因为静态收集的统计数据与真实评估样本的激活统计数据之间的差异。

在我们实际使用过程中，在 SmoothQuant 平滑后，对应的存储和计算精度都可以是 INT8 或者 FP8，可以利用 INT8 或者 FP8 的 TensorCore 进行计算。在实现细节上，权重支持 Per-tensor 和 Per-channel 的量化，激活值支持 Per-tensor 和 Per-token 的量化。

### SmoothQuant 的核心变换（数学等价的平滑）

```
原始：  Y = X · Wᵀ
变换后：Y = (X · diag(s)⁻¹) · (W · diag(s))ᵀ = X' · W'ᵀ

其中：
  X' = X / s  ← 激活除以 s，值变小了，更容易量化
  W' = W × s  ← 权重乘以 s，值变大了，但权重本来就容易量化

数学上：X'·W'ᵀ = (X/s)·(W×s)ᵀ = X·Wᵀ ✅ 完全等价
```

### s 是怎么算的？

> ⚠️ **易错校正：s 的公式（重要！）**
>
> 学习对话中曾用过一个公式 `s_j = 1-α×(max|X_j|/max|X|)+α`，**这不是 SmoothQuant 原论文公式**。历史推演（仅作对照，勿用于正式理解）：
>
> ```
> s₀ = 1-0.5×(2.0/7.0)+0.5 = 1.357
> s₁ = 1-0.5×(7.0/7.0)+0.5 = 1.0
> s₂ = 1-0.5×(0.8/7.0)+0.5 = 1.443
> s₃ = 1-0.5×(0.6/7.0)+0.5 = 1.457
> → s = [1.357, 1.0, 1.443, 1.457]
> → W'[out0] = [0.5×1.357, 0.1×1.0, 0.3×1.443, 6.0×1.457] = [0.679, 0.1, 0.433, 8.742]
> ```
>
> 这个公式的问题：它只用了 `max|X_j|/max|X|`（激活内部比较），**完全没有用到权重的幅值**，且 s₁=1.0 表示 c1（离群最大的通道）反而没被平滑——这与"平滑离群通道"的直觉相反。原论文公式是：
>
> ```
> s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
>
> 其中：
>   max(|X_j|)：calibration samples 上第 j 个 activation channel 的最大幅值
>   max(|W_j|)：与该 input channel 对应的 weight channel 最大幅值
>   α：把多少量化难度从 activation 迁移给 weight
> ```
>
> α=0.5 时：`s_j = sqrt(max|X_j| / max|W_j|)`，用贯穿数据算得 **s = [1.690, 2.789, 1.155, 0.316]**（见下文本体推演）。它同时考虑激活与权重两侧幅值，离群通道 c1 得到最大 s=2.789 被重点平滑，符合直觉。直观理解：**激活该通道离群值越大 → s 越大 → X/s 被压得越狠；同时 W×s 放大，把难度转移到本来就容易量化的权重上。**

### 用数据一步步推演 s 和平滑变换

**第 1 步：算每列的最大值**

```
激活 X 每列最大值：max|X| = [2.0, 7.0, 0.8, 0.6]
权重对应 input-channel 最大值：max|W[:,j]| = [0.7, 0.9, 0.6, 6.0]
```

**第 2 步：算每列的 s（α=0.5）**

```
s_j = sqrt(max|X_j| / max|W_j|)
s₀ = sqrt(2.0/0.7)  ≈ 1.690
s₁ = sqrt(7.0/0.9)  ≈ 2.789  ← c1 离群值 7.0 被压成 7.0/2.789≈2.51
s₂ = sqrt(0.8/0.6)  ≈ 1.155
s₃ = sqrt(0.6/6.0)  ≈ 0.316
```

**第 3 步：应用平滑变换**

```
W' 每列乘以对应 s：
out0': [0.5×1.690, 0.1×2.789, 0.3×1.155, 6.0×0.316]
      = [0.845, 0.279, 0.347, 1.896]
out1': [0.4×1.690, 0.2×2.789, 0.6×1.155, 0.8×0.316]
      = [0.676, 0.558, 0.693, 0.253]
...

X' 每列除以对应 s：
token0': [2.0/1.690, 0.5/2.789, 0.8/1.155, 0.3/0.316]
        = [1.183, 0.179, 0.693, 0.949]
token1': [0.4/1.690, 7.0/2.789, 0.2/1.155, 0.6/0.316]
        = [0.237, 2.510, 0.173, 1.899]
```

**观察**：c1 的激活离群值 7.0 被压成 2.51（离群被稀释），对应的权重 c1 列被放大（0.1→0.279），量化难度迁移给了原本更容易量化的权重。

**第 4 步：验证数学等价性**

```
平滑前：Y[0][0] = 2.0×0.5 + 0.5×0.1 + 0.8×0.3 + 0.3×6.0 = 3.09
平滑后（纯数学）：Y'[0][0] = (2.0/1.690)×(0.5×1.690) + ... = 3.09 ✅
```

**第 5 步：误差对比（平滑 vs 不平滑）**

```
不平滑直接量化 X（unscaled s=7.0/127=0.0551）：
  Y[0][0] = 2.946，误差 0.144 ❌

平滑后量化 X'+W'：
  Y'[0][0] = 3.105，误差 0.015 ✅
```

### 三种策略 O1/O2/O3

| 策略 | 权重量化 | 激活量化 | 计算效率 | 精度 |
|:----|:--------|:--------|:--------|:----|
| **O1** | per-channel INT8 | per-token INT8 | 慢（per-channel 不兼容 GEMM） | 最高 |
| **O2** | per-channel INT8 | per-tensor INT8 | 中等 | 好 |
| **O3** | per-tensor INT8 | per-tensor INT8 | 最快 | 下降 0.8% |

最常见的是 O2 和 O3。

> 🔍 **学习追问 1：SmoothQuant 是离线的还是在线的？用校准数据集了吗？**
>
> **离线的，需要校准集。** SmoothQuant 属于"离线静态量化"：
> ```
> 离线阶段（部署前，做一次）：
>   1. 准备校准数据集（如 512 条文本）
>   2. 跑一遍校准数据，收集每层激活的每列最大值
>   3. 算每列的平滑因子 sⱼ
>   4. 对 W 做变换 W'=W×s，量化 W'_int8，存 s 和 W'_int8
> 在线阶段（推理时）：
>   1. 拿到输入 X，直接用离线算好的 s 做变换：X'=X/s
>   2. 量化 X'，做 INT8 GEMM
> ```
> **s 是离线算好固定下来的，推理时直接用，不再重新算。** 对比 LLM.int8()（不需要校准集、在线动态检测），SmoothQuant 是离线静态量化。

> 🔍 **学习追问 2：推理时 X 是不是每列都现场除 s？**
>
> 数学表达确实是 `X_hat = X / s`，但 SmoothQuant 的关键工程技巧之一是：**尽可能把 smoothing factor 折叠进前一层参数中，离线完成等价变换**。因此实际 runtime 不一定需要单独 launch 一个 `X/s` kernel，runtime activation 本身已经被"平滑"了。这点非常重要，因为额外 elementwise kernel 会吃掉量化收益。

### SmoothQuant vs LLM.int8()

| | SmoothQuant | LLM.int8() |
|:--|:-----------|:----------|
| 思路 | 把激活的难度转移到权重 | 离群值单独拎出来用 FP16 |
| 计算 | 一个 INT8 GEMM 搞定 | INT8 GEMM + FP16 GEMM + 合并 |
| 推理效率 | 高 | 低（答主说"推理性能差"） |
| 是否需要校准 | 需要 | 不需要 |
| 实现复杂度 | 低（只需算 s 乘一下） | 高（实时检测、拆分、合并） |

> 🔍 **一句话总结 SmoothQuant**
>
> SmoothQuant 发现 LLM activation 的 outlier 集中在固定 channel，而 weight 更容易量化；它通过离线 per-input-channel 等价缩放 `X/s` 与 `W*s` 把 activation 的量化难度迁移到 weight，并用 `s_j=max|X_j|^α/max|W_j|^(1-α)` 平衡两者，从而实现硬件友好的 W8A8 INT8 GEMM。

---

**博主原文配图**（源自知乎原文）：

![原文图 8](https://pica.zhimg.com/v2-39a0d6788c139ea7a4b5e5f370dec1b4_r.jpg?source=2c26e567)


![原文图 9](https://picx.zhimg.com/v2-bb447a550edca7d11e68d4263b23bf0c_r.jpg?source=2c26e567)


![原文图 10](https://picx.zhimg.com/v2-564a8f9e94f078fe62372bc96d76c7df_r.jpg?source=2c26e567)


---

### 📌 原始 Q&A：SmoothQuant（历史第 60-66 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 60 轮）

那现在我们来讨论：SmoothQuant

##### A（历史第 61 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！用我们的 4×4 数据完整推演 **SmoothQuant**。

---

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

激活 X(2×4)：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]
```

---

###### 一、SmoothQuant 的核心问题

答主原文说：

> 针对大模型而言，**权重很容易量化，对于激活值，由于异常值的存在，导致激活则较难量化。**

看我们数据：

```
激活 X：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]

如果对激活做 per-tensor INT8 量化：
s = 7.0 / 127 = 0.0551
token0 的 [2.0, 0.5, 0.8, 0.3] 全部用粗步长 0.0551 → 精度差

如果对激活做 per-channel 量化：
精度好，但不兼容 INT8 GEMM Kernel

如果对激活做 per-token 量化：
精度好，token0 用 s=2.0/127=0.0157，但 token1 还是被 7.0 拖累
```

**SmoothQuant 的解法：既然激活难量化、权重容易量化，那就把激活中的"量化难度"转移到权重上。**

---

###### 二、核心思想：数学等价的平滑变换

SmoothQuant 引入一个平滑因子 s，做如下变换：

```
原始：  Y = X · Wᵀ

变换后：Y = (X · diag(s)⁻¹) · (W · diag(s))ᵀ
           = X' · W'ᵀ

其中：
  X' = X / s  ← 激活除以 s，值变小了，更容易量化
  W' = W × s  ← 权重乘以 s，值变大了，但权重本来就容易量化

数学上：X' · W'ᵀ = (X/s) · (W×s)ᵀ = X · Wᵀ ✅ 完全等价
```

---

###### 三、s 是怎么算的？

SmoothQuant 对每列算一个 s：

```
sⱼ = 1 - α × (max|Xⱼ| / max|X|) + α

其中：
  Xⱼ     = X 的第 j 列
  max|Xⱼ| = 第 j 列的最大绝对值
  max|X|  = 整个 X 的最大绝对值
  α      = 平滑强度（默认 0.5），控制"转移多少难度"
```

---

###### 四、用我们数据计算 s

###### 第 1 步：算每列的最大值

```
X:
         c0    c1    c2    c3
token0: [2.0,  0.5,  0.8,  0.3]
token1: [0.4,  7.0,  0.2,  0.6]

每列最大值：
  max|X₀| = max(|2.0|, |0.4|) = 2.0
  max|X₁| = max(|0.5|, |7.0|) = 7.0
  max|X₂| = max(|0.8|, |0.2|) = 0.8
  max|X₃| = max(|0.3|, |0.6|) = 0.6

整个 X 的最大值：
  max|X| = 7.0
```

###### 第 2 步：算每列的 s（α=0.5）

```
s₀ = 1 - 0.5 × (2.0/7.0) + 0.5 = 1 - 0.5×0.286 + 0.5 = 1 - 0.143 + 0.5 = 1.357
s₁ = 1 - 0.5 × (7.0/7.0) + 0.5 = 1 - 0.5×1.0 + 0.5 = 1 - 0.5 + 0.5 = 1.0
s₂ = 1 - 0.5 × (0.8/7.0) + 0.5 = 1 - 0.5×0.114 + 0.5 = 1 - 0.057 + 0.5 = 1.443
s₃ = 1 - 0.5 × (0.6/7.0) + 0.5 = 1 - 0.5×0.086 + 0.5 = 1 - 0.043 + 0.5 = 1.457

s = [1.357, 1.0, 1.443, 1.457]
```

**观察**：
- c1 列有最大离群值 7.0 → s₁=1.0（不缩放，因为已经够大了，不用转移）
- c0 列有 2.0 → s₀=1.357（s>1，激活除以 s 后变小，权重乘以 s 后变大）
- c2, c3 列值小 → s₂=1.443, s₃=1.457（s>1，转移更多难度到权重）

---

###### 五、应用平滑变换

###### 激活 X' = X / s

```
token0': [2.0/1.357, 0.5/1.0, 0.8/1.443, 0.3/1.457]
        = [1.474, 0.5, 0.554, 0.206]

token1': [0.4/1.357, 7.0/1.0, 0.2/1.443, 0.6/1.457]
        = [0.295, 7.0, 0.139, 0.412]
```

**平滑前后对比**：

```
平滑前 X：
token0: [2.0,  0.5,  0.8,  0.3]
token1: [0.4,  7.0,  0.2,  0.6]
         ↑ 2.0         ↑ 0.8   ← 范围跨度大

平滑后 X'：
token0: [1.474, 0.5, 0.554, 0.206]
token1: [0.295, 7.0, 0.139, 0.412]
         ↑ 变小了！     ↑ 变小了！
```

**c0 列从 2.0 降到了 1.474，c2 列从 0.8 降到了 0.554，c3 列从 0.3/0.6 降到了 0.206/0.412。激活的离群值被压了。**

###### 权重 W' = W × s

```
W' 每列乘以对应的 s：

out0': [0.5×1.357, 0.1×1.0, 0.3×1.443, 6.0×1.457]
      = [0.679, 0.1, 0.433, 8.742]
out1': [0.4×1.357, 0.2×1.0, 0.6×1.443, 0.8×1.457]
      = [0.543, 0.2, 0.866, 1.166]
out2': [0.7×1.357, 0.3×1.0, 0.5×1.443, 0.2×1.457]
      = [0.950, 0.3, 0.722, 0.291]
out3': [0.1×1.357, 0.9×1.0, 0.4×1.443, 0.6×1.457]
      = [0.136, 0.9, 0.577, 0.874]
```

**变换前后对比**：

```
平滑前 W：
out0: [0.5, 0.1, 0.3, 6.0]
                      ↑ 6.0

平滑后 W'：
out0: [0.679, 0.1, 0.433, 8.742]
                          ↑ 8.742！权重更难量化了！
```

**权重确实更难量化了（6.0→8.742），但权重本来就容易量化，INT8 的 127 个格子完全覆盖 8.742，精度损失不大。**

---

###### 六、平滑后再做 INT8 量化

###### 量化 X'

对平滑后的 X' 做 per-token INT8 量化：

```
token0': [1.474, 0.5, 0.554, 0.206]
  s_token0 = 1.474/127 = 0.0116
  1.474→127, 0.5→43, 0.554→48, 0.206→18

token1': [0.295, 7.0, 0.139, 0.412]
  s_token1 = 7.0/127 = 0.0551
  0.295→5, 7.0→127, 0.139→3, 0.412→7
```

对比平滑前直接量化 token0：

```
平滑前量化 token0：s=2.0/127=0.0157
  token0 的小值（0.3, 0.5, 0.8）用 0.0157 的步长

平滑后量化 token0：s=1.474/127=0.0116
  token0 的小值用 0.0116 的步长 ← 更精细了！
```

**这就是 SmoothQuant 的收益：激活的 range 被压缩了，步长更细，精度更高。**

###### 量化 W'

对平滑后的 W' 做 per-channel INT8 量化：

```
out0': [0.679, 0.1, 0.433, 8.742]
  s_out0 = 8.742/127 = 0.0688
  0.679→10, 0.1→1, 0.433→6, 8.742→127

out1': [0.543, 0.2, 0.866, 1.166]
  s_out1 = 1.166/127 = 0.0092
  0.543→59, 0.2→22, 0.866→94, 1.166→127
```

###### 做 INT8 矩阵乘

```
Y = INT8_GEMM(X'_int8, W'_int8ᵀ) → 反量化
```

这一步就是一个干净利落的 INT8 矩阵乘，**不需要拆分矩阵，不需要混合精度**。

---

###### 七、验证数学等价性

```
平滑前：
Y[0][0] = 2.0×0.5 + 0.5×0.1 + 0.8×0.3 + 0.3×6.0
        = 1.0 + 0.05 + 0.24 + 1.8
        = 3.09

平滑后（反量化前，纯数学）：
Y'[0][0] = 1.474×0.679 + 0.5×0.1 + 0.554×0.433 + 0.206×8.742
         = 1.0 + 0.05 + 0.24 + 1.8
         = 3.09 ✅ 数学上完全等价
```

但如果用了 INT8 量化后会有舍入误差，但比不平滑直接量化要好得多。

---

###### 八、对比：不平滑直接 INT8 vs 平滑后 INT8

###### 不平滑直接量化 X

```
unscaled s = 7.0/127 = 0.0551

token0: [2.0, 0.5, 0.8, 0.3]
  2.0→36→1.984, 0.5→9→0.496, 0.8→15→0.827, 0.3→5→0.276

Y[0][0] = 1.984×0.5 + 0.496×0.1 + 0.827×0.3 + 0.276×6.0
        = 0.992 + 0.050 + 0.248 + 1.656
        = 2.946

误差 = 3.09 - 2.946 = 0.144
```

###### 平滑后量化 X' + W'

```
平滑后 s_token0 = 1.474/127 = 0.0116
token0': [1.474, 0.5, 0.554, 0.206]
  1.474→127→1.474, 0.5→43→0.499, 0.554→48→0.557, 0.206→18→0.209

out0': [0.679, 0.1, 0.433, 8.742]
  s_out0 = 8.742/127 = 0.0688
  0.679→10→0.688, 0.1→1→0.069, 0.433→6→0.413, 8.742→127→8.742

Y'[0][0] = 1.474×0.688 + 0.499×0.069 + 0.557×0.413 + 0.209×8.742
         = 1.014 + 0.034 + 0.230 + 1.827
         = 3.105

误差 = 3.105 - 3.09 = 0.015 ✅ 比 0.144 好多了
```

---

###### 九、三种策略

答主原文说的三种策略：

| 策略 | 权重量化 | 激活量化 | 计算效率 | 精度 |
|:----|:--------|:--------|:--------|:----|
| **O1** | per-channel INT8 | per-token INT8 | 慢（per-channel 不兼容 GEMM） | 最高 |
| **O2** | per-channel INT8 | per-tensor INT8 | 中等 | 好 |
| **O3** | per-tensor INT8 | per-tensor INT8 | 最快 | 下降 0.8% |

最常见的是 O2 和 O3。

---

###### 十、SmoothQuant vs LLM.int8()

| | SmoothQuant | LLM.int8() |
|:--|:-----------|:----------|
| 思路 | 把激活的难度转移到权重 | 离群值单独拎出来用 FP16 |
| 计算 | 一个 INT8 GEMM 搞定 | INT8 GEMM + FP16 GEMM + 合并 |
| 推理效率 | 高 | 低（答主说"推理性能差"） |
| 是否需要校准 | 需要 | 不需要 |
| 实现复杂度 | 低（只需算 s 乘一下） | 高（实时检测、拆分、合并） |

---

###### 十一、一句话总结

> **SmoothQuant 通过给每列算一个平滑因子 s，让激活除以 s（变小、易量化）、权重乘以 s（变大、但权重本身皮实），把量化难度从激活转移到权重上。变换后直接做 INT8 矩阵乘，不需要拆分、不需要混合精度。数学上等价于原始计算。**

##### Q（历史第 62 轮）

所以这部分的话其实 是 离线的还是在线的？用校准数据集了吗？

##### A（历史第 63 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好问题！

---

###### SmoothQuant 的 s 是**离线**算的，用了**校准集**

答主原文：

> **离线静态量化**：模型上线推理前使用校准集生成缩放因子，对权重和激活进行量化

SmoothQuant 属于这里的"离线静态量化"。

---

###### 用我们的数据看流程

###### 离线阶段（部署前，做一次）

```
1. 准备校准数据集（比如 512 条文本）

2. 跑一遍校准数据，收集每层激活的统计信息：
   每列的最大值 max|Xⱼ|
   全局最大值 max|X|

3. 算每列的平滑因子 sⱼ：
   sⱼ = 1 - α × (max|Xⱼ| / max|X|) + α

4. 算完 s 后，对 W 做变换：W' = W × s
   然后量化 W'：W'_int8

5. 把 s 和 W'_int8 存下来
```

###### 在线阶段（推理时）

```
1. 拿到输入 X
2. 直接用离线算好的 s 做变换：X' = X / s
3. 量化 X'：X'_int8
4. 做 INT8 GEMM：Y = X'_int8 · W'_int8ᵀ
```

**s 是离线算好固定下来的，推理时直接用，不再重新算。**

---

###### 对比 LLM.int8() 就清楚了

| | SmoothQuant | LLM.int8() |
|:--|:-----------|:----------|
| **用校准集吗？** | ✅ 需要，用来算 s | ❌ 不需要，加载时直接量化 |
| **s/阈值什么时候定？** | 离线定好 | 阈值 6.0 是论文定的固定值 |
| **推理时做什么？** | 直接用 s 做除法（固定操作） | **动态检测**离群值（每次可能不同） |
| **属于什么？** | 离线静态量化 | 在线动态量化 |

---

所以 SmoothQuant 是 **离线静态量化**，依赖校准集算出 s 后固定下来。

##### Q（历史第 64 轮）

奥奥原来是这样。就是说，根据校准的数据集，然后跑中间的激活数值。然后获取s，然后把W扩大S，然后进行量化，然后存储s和量化后的w

##### A（历史第 65 轮，正文原样保留；仅将内部标题降级以保持目录层级）

**完全正确！** 你的理解非常准确。

就是这几步：

```
1. 校准集 → 跑前向 → 收集每层激活的每列最大值
2. 算每列的 s
3. W' = W × s（每列乘对应的 s）
4. 量化 W' → W'_int8，存 W'_int8 + s
5. 推理时：X' = X / s → 量化 → INT8 GEMM
```

**补充一个关键细节**：s 是 **per-column（per-input-channel）** 的，不是 per-row 也不是 per-tensor。

```
W(4×4) 乘以 s(1×4)，每列乘对应的 s：
         c0    c1    c2    c3
        s₀=1.357  s₁=1.0  s₂=1.443  s₃=1.457

out0: [0.5×1.357, 0.1×1.0, 0.3×1.443, 6.0×1.457]
out1: [0.4×1.357, 0.2×1.0, 0.6×1.443, 0.8×1.457]
out2: [0.7×1.357, 0.3×1.0, 0.5×1.443, 0.2×1.457]
out3: [0.1×1.357, 0.9×1.0, 0.4×1.443, 0.6×1.457]
```

在推理时，X 进来也是每列除以对应的 s，然后量化。

**所以 SmoothQuant 的"离线工作"就是：算好 s → 乘到 W 上 → 量化 W → 存起来。推理时直接用，非常轻量。**

需要继续看 **AWQ** 吗？

##### Q（历史第 66 轮）

s是怎么算的，给我用那个数据讲述下

##### A（历史第 67 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！用我们的数据一步步算 **s** 的值。

---

###### 设定

```
激活 X(2×4)：
         c0    c1    c2    c3
token0: [2.0,  0.5,  0.8,  0.3]
token1: [0.4,  7.0,  0.2,  0.6]
```

---

###### 公式

```
sⱼ = 1 - α × (max|Xⱼ| / max|X|) + α

其中：
  Xⱼ      = X 的第 j 列（比如 c0 列）
  max|Xⱼ| = 第 j 列中最大的绝对值
  max|X|  = 整个 X 中最大的绝对值
  α       = 平滑强度，默认 0.5
```

---

###### 第 1 步：算每列的最大值

```
c0 列：token0=2.0, token1=0.4 → max|c0| = max(2.0, 0.4) = 2.0
c1 列：token0=0.5, token1=7.0 → max|c1| = max(0.5, 7.0) = 7.0
c2 列：token0=0.8, token1=0.2 → max|c2| = max(0.8, 0.2) = 0.8
c3 列：token0=0.3, token1=0.6 → max|c3| = max(0.3, 0.6) = 0.6

max|X| = max(2.0, 7.0, 0.8, 0.6) = 7.0
```

---

###### 第 2 步：算每列的比例

```
c0: max|c0| / max|X| = 2.0 / 7.0 = 0.286
c1: max|c1| / max|X| = 7.0 / 7.0 = 1.0
c2: max|c2| / max|X| = 0.8 / 7.0 = 0.114
c3: max|c3| / max|X| = 0.6 / 7.0 = 0.086
```

---

###### 第 3 步：代入公式（α=0.5）

```
s₀ = 1 - 0.5 × 0.286 + 0.5
   = 1 - 0.143 + 0.5
   = 1.357

s₁ = 1 - 0.5 × 1.0 + 0.5
   = 1 - 0.5 + 0.5
   = 1.0

s₂ = 1 - 0.5 × 0.114 + 0.5
   = 1 - 0.057 + 0.5
   = 1.443

s₃ = 1 - 0.5 × 0.086 + 0.5
   = 1 - 0.043 + 0.5
   = 1.457

s = [1.357, 1.0, 1.443, 1.457]
```

---

###### 第 4 步：换 α=0.7 看看有什么不同

```
α 越大，转移的难度越多：

s₀ = 1 - 0.7 × 0.286 + 0.7 = 1 - 0.200 + 0.7 = 1.500
s₁ = 1 - 0.7 × 1.0 + 0.7   = 1 - 0.7 + 0.7   = 1.0
s₂ = 1 - 0.7 × 0.114 + 0.7 = 1 - 0.080 + 0.7 = 1.620
s₃ = 1 - 0.7 × 0.086 + 0.7 = 1 - 0.060 + 0.7 = 1.640
```

α=0.7 时 s 更大，意味着 **更多难度从激活转移到权重**（X/s 变得更小，W×s 变得更大）。

---

###### 第 5 步：直观理解 s 的物理意义

```
s = [1.357, 1.0, 1.443, 1.457]

c1 的 s₁=1.0：   c1 本身有最大离群值 7.0，不缩放（已经是难度最大的了）
c0 的 s₀=1.357：  c0 有 2.0，中等大小，转移一部分难度到权重
c2 的 s₂=1.443：  c2 只有 0.8，转移较多难度到权重
c3 的 s₃=1.457：  c3 只有 0.6，转移最多难度到权重
```

**越小的列，s 越大，转移越多难度到权重。越大的列，s 越接近 1，少转移。**

---

###### 第 6 步：s 的数学范围

s 的值域是 **[1-α, 1+α-α×min_ratio, ...]**，直观来看：

```
当 max|Xⱼ| = max|X|（该列是全局最大）→ 比值=1 → s = 1 - α + α = 1
当 max|Xⱼ| = 0（该列全是 0）       → 比值=0 → s = 1 + α
所以 s 的取值范围是 [1, 1+α]
α=0.5 时：s ∈ [1.0, 1.5]
α=0.7 时：s ∈ [1.0, 1.7]
```

**s 永远 ≥ 1**，所以：
- X/s ≤ X（激活永远不会变大，只会变小或不变）→ 更容易量化
- W×s ≥ W（权重永远不会变小，只会变大或不变）→ 更难量化，但权重扛得住

---

###### 一句话

> **s 就是用"激活每列的最大值占全局最大值的比例"算出来的一个 ≥1 的缩放因子。比例越大（离群值越大），s 越接近 1（不转移）；比例越小（值越小），s 越大（转移更多难度到权重）。**

##### Q（历史第 68 轮）


---
假设，现在激活的每一列都算出来了s，那W是怎么扩大的呢？；在推理的过程中也是激活的每一列都/对应的列的s吗

##### A（历史第 69 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好问题！先说结论：

**W 扩大：W 的每列 × s 的对应列**
**推理时：X 的每列 ÷ s 的对应列**

用我们的数据画出来就清楚了。

---

###### s 是 per-column 的

```
s = [1.357, 1.0, 1.443, 1.457]
      ↑     ↑     ↑     ↑
     c0列  c1列  c2列  c3列
```

**s 有 4 个值，对应 4 个输入通道（列）。**

---

###### W 怎么扩大的？

W 的形状是 4×4，**每列**乘以对应列的 s：

```
W(4×4) × s(1×4) = W'(4×4)
                      ↑
               每列乘对应的 s

         c0         c1         c2         c3
        s₀=1.357   s₁=1.0     s₂=1.443   s₃=1.457

out0: [0.5×1.357, 0.1×1.0,   0.3×1.443, 6.0×1.457]  = [0.679, 0.1,  0.433, 8.742]
out1: [0.4×1.357, 0.2×1.0,   0.6×1.443, 0.8×1.457]  = [0.543, 0.2,  0.866, 1.166]
out2: [0.7×1.357, 0.3×1.0,   0.5×1.443, 0.2×1.457]  = [0.950, 0.3,  0.722, 0.291]
out3: [0.1×1.357, 0.9×1.0,   0.4×1.443, 0.6×1.457]  = [0.136, 0.9,  0.577, 0.874]
```

**每个 out 行的 c0 列都乘 s₀，c1 列都乘 s₁，依此类推。**

---

###### 推理时 X 怎么除的？

```
X(2×4) ÷ s(1×4) = X'(2×4)
                      ↑
               每列除以对应的 s

         c0         c1         c2         c3
        s₀=1.357   s₁=1.0     s₂=1.443   s₃=1.457

token0: [2.0/1.357, 0.5/1.0,  0.8/1.443, 0.3/1.457] = [1.474, 0.5,  0.554, 0.206]
token1: [0.4/1.357, 7.0/1.0,  0.2/1.443, 0.6/1.457] = [0.295, 7.0,  0.139, 0.412]
```

**每个 token 的 c0 列都除 s₀，c1 列都除 s₁，依此类推。**

---

###### 为什么数学上等价？

```
原始：  Y = X · Wᵀ

变换后：
Y' = X' · W'ᵀ

X'[i][j] = X[i][j] / s[j]    ← 第 j 列除以 sⱼ
W'[k][j] = W[k][j] × s[j]    ← 第 j 列乘以 sⱼ

Y'[i][k] = Σⱼ X'[i][j] × W'[k][j]
         = Σⱼ (X[i][j] / s[j]) × (W[k][j] × s[j])
         = Σⱼ X[i][j] × W[k][j] × (s[j] / s[j])        ← sⱼ 抵消了！
         = Σⱼ X[i][j] × W[k][j]
         = Y[i][k] ✅
```

**s 在每列内部自己乘自己除，抵消了。**

---

###### 一句话

> **s 是按列算的。W 的每列乘 s，X 的每列除 s，而且乘和除用的是同一个 sⱼ。在矩阵乘里 s 被抵消了，数学上完全等价。**

## 2.7 AWQ

**特点**：W4A16，per-group 量化（INT4-g128），量化速度慢

AWQ（AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration）方法源于"权重对于 LLM 的性能并不同等重要"的观察，存在约（0.1%-1%）显著权重对大模型性能影响太大，通过跳过这 1% 的重要权重（salient weight）不进行量化，可以大大减少量化误差。

尽管我们只做了权重量化，但要找到显著的权重通道，我们应该根据**激活分布**而不是权重分布——与较大激活幅度相对应的权重通道更加突出，因为它们处理了更重要的特征。

### AWQ 的核心思想

对显著权重引入缩放因子 s，进行等价变换：

```
Q(w·s) · (x/s)
```

即 w 乘以 s（s>1），x 除以 s。最终量化误差比值约为 Δ'/Δ × 1/s，显著权重的相对误差较小。

### s 的计算

```
s = s_X^α
```

其中 s_X 是激活的平均幅值（每个通道），α 是单个超参数，通过在 [0,1] 区间内网格搜索找到最佳值（0 表示不缩放，1 对应最激进的缩放）。

优化目标：
```
s* = arg_s min ||Q(W·diag(s))(diag(s)^{-1}·X) - WX||
```

### 用数据一步步推演 AWQ

**第 1 步：找显著通道（根据激活分布）**

```
X 的每列平均幅值（绝对值）：
  c0: (|2.0|+|0.4|)/2 = 1.2
  c1: (|0.5|+|7.0|)/2 = 3.75  ← 最大！c1 是最显著通道
  c2: (|0.8|+|0.2|)/2 = 0.5
  c3: (|0.3|+|0.6|)/2 = 0.45

c1 的激活幅值最大 → c1 对应的权重通道最显著
```

**第 2 步：计算缩放因子 s_X = 激活每列的平均幅值 = [1.2, 3.75, 0.5, 0.45]**

**第 3 步：用 grid search 找最佳 α**

```
s = s_X^α，α ∈ [0,1]
α=0 时：s=[1,1,1,1] → 不做缩放，等于 RTN
α=0.5 时：s = [1.2^0.5, 3.75^0.5, 0.5^0.5, 0.45^0.5] = [1.095, 1.936, 0.707, 0.671]
α=1.0 时：s = [1.2, 3.75, 0.5, 0.45] → 最激进

在 [0,1] 区间等间隔采样（比如 20 个值），对每个 α 算量化后输出误差，选误差最小的 α。
```

> 🔍 **为什么不用梯度法搜索 α，而用 grid search？**
>
> 博主原文：有一些技术依赖于近似梯度，但**发现它仍然存在收敛不稳定的问题**。为了使这一过程更加稳定，作者通过分析影响缩放因子选择的因素，为最佳缩放比例**定义了一个搜索空间**（即 `s = s_X^α`，α∈[0,1]），然后做快速网格搜索（grid search）。0 表示不做缩放，1 对应最激进的缩放。

**第 4 步：应用缩放（假设 α=0.5）**

```
W' = W × diag(s)：
   out0: [0.5×1.095, 0.1×1.936, 0.3×0.707, 6.0×0.671]
        = [0.548, 0.194, 0.212, 4.026]
   out1: [0.4×1.095, 0.2×1.936, 0.6×0.707, 0.8×0.671]
        = [0.438, 0.387, 0.424, 0.537]
   out2: [0.7×1.095, 0.3×1.936, 0.5×0.707, 0.2×0.671]
        = [0.767, 0.581, 0.354, 0.134]
   out3: [0.1×1.095, 0.9×1.936, 0.4×0.707, 0.6×0.671]
        = [0.110, 1.742, 0.283, 0.403]

X' = X / diag(s)：
   token0: [2.0/1.095, 0.5/1.936, 0.8/0.707, 0.3/0.671]
          = [1.826, 0.258, 1.132, 0.447]
   token1: [0.4/1.095, 7.0/1.936, 0.2/0.707, 0.6/0.671]
          = [0.365, 3.616, 0.283, 0.894]
```

**观察**：c1（显著通道）s>1 → W 被放大 → 量化时更精确（值越大相对误差越小）；c2,c3（非显著通道）s<1 → W 被缩小 → 量化精度略降，但反正是非显著通道，影响小。

### AWQ vs SmoothQuant

| 维度 | AWQ | SmoothQuant |
|---|---|---|
| 目标 | W4A16（仅权重量化） | W8A8（权重+激活量化） |
| s 的来源 | 激活的平均幅值 s_X | max\|X_j\| / max\|W_j\| 的比值 |
| s 的公式 | s = s_X^α（平均幅值的 α 次方） | s = max\|X_j\|^α / max\|W_j\|^(1-α) |
| s 的确定 | 网格搜索 α | 公式直接计算（α 可调） |
| 激活处理 | 不量化激活（A16） | 量化激活（A8） |
| s 的范围 | 可大于1也可小于1 | 可大于1也可小于1 |
| 显著通道策略 | **有偏向地保护**显著通道，牺牲非显著通道 | 均匀地"压激活、抬权重"，没有偏向 |

> ⚠️ **易错校正："保护 1% 显著权重"到底是什么意思？**
>
> AWQ 论文的动机实验发现：如果只把极少数 salient weights 保持高精度，误差会明显下降。但最终 AWQ 的硬件友好方案**不是简单永久保留 1% FP16 weights**（那会造成 irregular mixed precision，kernel 很难高效）。真正方案是：**通过 per-channel scaling 放大重要 weight channel，然后再统一做低 bit quantization**，这样重要 channel 相对于量化 step 有更高有效分辨率。

> 🔍 **AWQ 的另一个技巧：权重剪裁（weight clipping）**
>
> 除了逐通道缩放，AWQ 还**通过应用权重剪裁来进一步最小化 MSE 误差**——因为剪裁（clipping）权重可以进一步帮助减少 `Q(w·s)·(x/s)` 与原始输出的差异。做法：对缩放后的权重做截断，限制其范围，让量化 step 更集中。这和 SmoothQuant/GPTQ 中"clip 减少截断误差"的思想一致，属于 AWQ 论文中的补充优化。

> 🔍 **AWQ 面试一句话**
>
> AWQ 是 activation-aware 的 W4 weight-only PTQ，它用 calibration activation 判断哪些 input channels 更重要，然后通过等价 per-channel scaling 放大这些通道的 weights，降低其相对量化误差，同时避免真正的 FP16/INT4 irregular mixed-precision weight layout。

---

**博主原文配图**（源自知乎原文）：

![原文图 11](https://picx.zhimg.com/v2-14335c361196cf8031c73505458cb46d_r.jpg?source=2c26e567)


---

### 📌 原始 Q&A：AWQ（历史第 70 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 70 轮）

接下来我们讨论这部分：AWQ

##### A（历史第 71 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！用我们的 4×4 数据完整推演 **AWQ**。

---

###### 设定回顾

```
权重 W(4×4)：
out0: [0.5, 0.1, 0.3, 6.0]
out1: [0.4, 0.2, 0.6, 0.8]
out2: [0.7, 0.3, 0.5, 0.2]
out3: [0.1, 0.9, 0.4, 0.6]

激活 X(2×4)：
token0: [2.0, 0.5, 0.8, 0.3]
token1: [0.4, 7.0, 0.2, 0.6]
```

---

###### 一、AWQ 的核心思想

答主原文：

> **AWQ 方法源于"权重对于 LLM 的性能并不同等重要"的观察，存在约（0.1%-1%）显著权重对大模型性能影响太大。**

**核心：不是所有权重同等重要。那些对应大激活值的权重通道更重要，需要被保护。**

而且找显著通道的依据是 **激活分布**，不是权重分布：

> 尽管我们只做了权重量化，但要找到显著的权重通道，我们应该根据**激活分布**而不是权重分布。

---

###### 二、AWQ 和 SmoothQuant 的相似性

AWQ 和 SmoothQuant 一样，都是：

```
W' = W × s
X' = X / s
```

然后做 RTN 量化。但 **s 的计算方式完全不同**。

---

###### 三、s 的来源不同

```
SmoothQuant 的 s：
  sⱼ = 1 - α × (max|Xⱼ| / max|X|) + α
  s 由"列的最大值"决定

AWQ 的 s：
  sⱼ = (s_Xⱼ)^α
  其中 s_Xⱼ 是"激活第 j 列的平均幅值"
  α 由 grid search 搜索得到
```

---

###### 四、第 1 步：算激活每列的平均幅值 s_X

```
X:
         c0    c1    c2    c3
token0: [2.0,  0.5,  0.8,  0.3]
token1: [0.4,  7.0,  0.2,  0.6]

s_Xⱼ = (|token0ⱼ| + |token1ⱼ|) / 2

s_X₀ = (|2.0| + |0.4|) / 2 = 2.4 / 2 = 1.2
s_X₁ = (|0.5| + |7.0|) / 2 = 7.5 / 2 = 3.75  ← 最大！c1 是最显著通道
s_X₂ = (|0.8| + |0.2|) / 2 = 1.0 / 2 = 0.5
s_X₃ = (|0.3| + |0.6|) / 2 = 0.9 / 2 = 0.45

s_X = [1.2, 3.75, 0.5, 0.45]
```

**c1 的 s_X₁=3.75 最大 → c1 是最显著的通道，对应的权重通道需要被保护。**

---

###### 五、第 2 步：s = (s_X)^α

```
sⱼ = (s_Xⱼ)^α

α 在 [0, 1] 之间用 grid search 搜索。
```

###### 尝试不同的 α

```
α=0（不缩放）：
  s = [1.2⁰, 3.75⁰, 0.5⁰, 0.45⁰] = [1, 1, 1, 1]
  → 等价于 RTN，不做任何保护

α=0.5：
  s = [1.2⁰·⁵, 3.75⁰·⁵, 0.5⁰·⁵, 0.45⁰·⁵]
    = [1.095, 1.936, 0.707, 0.671]

α=1.0（最激进）：
  s = [1.2¹, 3.75¹, 0.5¹, 0.45¹] = [1.2, 3.75, 0.5, 0.45]
```

---

###### 六、第 3 步：应用 s（假设 α=0.5）

###### W' = W × s

```
s = [1.095, 1.936, 0.707, 0.671]

out0': [0.5×1.095, 0.1×1.936, 0.3×0.707, 6.0×0.671]
      = [0.548, 0.194, 0.212, 4.026]
out1': [0.4×1.095, 0.2×1.936, 0.6×0.707, 0.8×0.671]
      = [0.438, 0.387, 0.424, 0.537]
out2': [0.7×1.095, 0.3×1.936, 0.5×0.707, 0.2×0.671]
      = [0.767, 0.581, 0.354, 0.134]
out3': [0.1×1.095, 0.9×1.936, 0.4×0.707, 0.6×0.671]
      = [0.110, 1.742, 0.283, 0.403]
```

**c1（显著通道，s=1.936>1）：权重被放大，量化时更精确！**
**c2, c3（非显著通道，s=0.707, 0.671<1）：权重被缩小，量化精度稍微牺牲。**

###### X' = X / s

```
s = [1.095, 1.936, 0.707, 0.671]

token0': [2.0/1.095, 0.5/1.936, 0.8/0.707, 0.3/0.671]
        = [1.826, 0.258, 1.132, 0.447]
token1': [0.4/1.095, 7.0/1.936, 0.2/0.707, 0.6/0.671]
        = [0.365, 3.616, 0.283, 0.894]
```

**c1（s=1.936）：激活被缩小最多 → 量化时 c1 列的误差以 (1/s) 缩小，相当于保护了 c1 通道的精度。**

---

###### 七、α 怎么找？

答主原文：

> 仅使用单个超参数 α 来平衡显著通道和非显著通道的保护。我们通过在 [0, 1] 区间内进行快速网格搜索（grid search）来找到最佳的 α。

```
在 [0, 1] 区间内等间隔采样（比如 20 个值）：
  α = 0, 0.05, 0.1, 0.15, ..., 1.0

对每个 α：
  1. 算 s = (s_X)^α
  2. 做 W' = W × s, X' = X / s
  3. 量化 W' 到 INT4
  4. 算量化后输出的误差

选误差最小的 α
```

---

###### 八、AWQ vs SmoothQuant

###### s 的计算对比

| | SmoothQuant | AWQ |
|:--|:-----------|:----|
| **s 的公式** | sⱼ = 1 - α×(max|Xⱼ|/max|X|) + α | sⱼ = (s_Xⱼ)^α |
| **依据** | 列的最大值 | 列的平均幅值 |
| **α 怎么选** | 固定（默认 0.5） | **grid search 搜索最优** |
| **s 的范围** | s ∈ [1, 1+α]（永远 ≥1） | s = (s_X)^α，可大于1也可小于1 |
| **显著通道的策略** | 不缩放或略微缩小 | **放大权重（s>1），缩小激活（÷s）** |

###### 关键区别

**SmoothQuant 的 s ≥ 1**：激活永远变小或不变，权重永远变大或不变。目的是让激活更容易量化。

**AWQ 的 s 取决于 s_X**：
- 显著通道（s_X 大）：s>1，权重大幅放大，激活大幅缩小 → **保护显著通道**
- 非显著通道（s_X 小）：s<1，权重缩小，激活放大 → **牺牲非显著通道**

```
SmoothQuant：均匀地"压激活、抬权重"，没有偏向
AWQ：有偏向地保护显著通道，牺牲非显著通道
```

---

###### 九、答主原文的关键句

> 我们通过分析影响缩放因子选择的因数，为最佳缩放比例定义了一个搜索空间。
> s = s_X^α
> α* = arg_α min L(s_X^α)

> 其中，s 仅与激活 s_X 的大小有关，s_X 是激活的平均幅值（每个通道），这里仅使用单个超参数 α 来平衡显著通道和非显著通道的保护。

---

###### 十、AWQ 为什么针对 W4A16？

**AWQ 是仅权重量化（W4A16）**，不是权重激活量化（W8A8）。

所以 AWQ 的流程是：

```
离线：
  1. 校准集 → 算每列平均幅值 s_X
  2. grid search 找最优 α
  3. 算 s = (s_X)^α
  4. W' = W × s
  5. 量化 W' 到 INT4 → W_int4
  6. 存 W_int4 + s

推理时：
  1. X' = X / s（激活做除法）
  2. W_int4 → 反量化 → W'_fp16
  3. Y = X' · W'_fp16ᵀ
```

> 注意 AWQ 计算时激活是 FP16（W4A16），不是 INT8。SmoothQuant 才是 W8A8。

## 2.8 SpQR

**特点**：仅权重量化、双层量化（对量化元数据进一步进行量化）、非对称（最小-最大）量化

之前的 LLM 量化算法同等对待低敏感度权重和高敏感度权重，这可能会导致次优量化。

SpQR 提出隔离此类异常值，并以 SpQR 格式有效地对给定模型进行编码。为了利用所得结构，还开发了一种基于压缩稀疏行（CSR）格式的专门稀疏矩阵乘法算法。为了使用 SpQR 进行逐个 token 生成，将这种稀疏算法与 3-4 比特权重的密集量化矩阵乘法结合起来。与 LLM 生成 16 比特推理相比，SpQR 将 LLMs 的内存占用减少了约 3.4 倍或更多，从而实现小于 1% 的精度损失。

### 用数据推演 SpQR

```
第 1 步：找离群权重
  W 中检测离群值：out0 的 6.0 是离群值
  把 6.0 隔离出来，单独存为 FP16（用稀疏矩阵存）
  剩下的非离群值用 3bit 量化

第 2 步：双层量化
  分组量化（group=16），然后对 scale 再做一次量化：
    原始 scale：FP16
    再量化 scale：3bit
```

### 工作原理

1. **确定并隔离离群权重**：对其量化会导致不成比例的高误差。将这些权重保持高精度，使用一个稀疏矩阵单独保存；而其他权重存储在低得多的精度中（如 3 比特格式）。
2. **细粒度分组量化**：重要参数往往以行或列聚集，实现一种具有非常小的组大小（如 16 个连续元素）的分组量化变体。同时，将量化缩放（scales）本身量化为 3 比特表示（压缩量化元数据，即双层量化）。

---

## 2.9 ZeroQuant 系列

### ZeroQuant

**特点**：W8A8、对权重矩阵使用分组量化、对激活进行逐 Token 动态量化，不需要校准激活范围

- 对权重矩阵进行分组量化，将整个权重矩阵分成 64 或 128 组。设计时考虑了 GPU Ampere 架构（如 A100）的硬件约束，计算单元基于 WMMA 的分片大小。
- 对激活采用逐 Token 量化并动态计算每个 Token 的最小/最大范围。
- 构建高度优化的推理后端：采用 Kernel 融合技术将量化算子与其先前的算子（如层归一化）融合，减轻逐 Token 量化的数据移动成本。

> 🔍 **博主原文补充（输出侧融合）**：逐 Token 量化引入了额外的量化和反量化操作，会导致 GPU 计算单元和主存之间产生昂贵的数据移动开销。ZeroQuant 的融合是双向的：
> 1. **输入侧**：量化算子与前一个算子（如 LayerNorm）融合；
> 2. **输出侧**：在将最终 FP16 结果写回到下一个 FP16 算子（如 GeLU）的主存之前，**使用权重和激活量化缩放因子直接缩放 INT32 累加**，从而减轻不同 GEMM 输出的反量化成本。
>
> 这样 INT8 GEMM 的 INT32 累加结果在片上用 scale 缩回 FP16，而不是每层都额外 launch 反量化 kernel。

**用数据推演（分组权重 + 逐 token 动态激活）**：

```
权重分组量化（group=2）：
  out0: [0.5, 0.1 | 0.3, 6.0]；out1: [0.4, 0.2 | 0.6, 0.8]；...

激活逐 token 动态量化（推理时实时算）：
  token0: [2.0, 0.5, 0.8, 0.3] → 动态算 s=2.0/127=0.0157 → [127, 32, 51, 19]
  token1: [0.4, 7.0, 0.2, 0.6] → 动态算 s=7.0/127=0.0551 → [7, 127, 4, 11]

Kernel 融合：把量化操作融合到前一个算子（如 LayerNorm）中，减少数据移动开销。
```

### ZeroQuant-V2

**特点**：新增了模型参数，拖累了推理的性能

引入**低秩补偿（LoRC）**：通过新增两个低秩矩阵来近似量化误差，以最小的模型参数大小增加来提升模型质量的恢复。

```
Y = X · W_int4_reconᵀ + X · (A·B)ᵀ
    ↑ 量化部分           ↑ 低秩补偿
A 和 B 很小，也可以量化为 8bit
```

### ZeroQuant-FP

**特点**：W4A8、FP4、FP8

- 发现对于 LLMs，**FP8 激活始终优于 INT8 激活**，且在参数超过 10 亿的模型中性能优势更明显。
- 对于权重量化，**FP4 表现出与 INT4 相当的性能**，简化了在 H100 等支持 FP 的硬件上的部署。
- 提出**位移（bit-shifting）方法**：将 scale 限制为 2 的幂（S=2ⁿ），简化 FP4→FP8 的转换，避免直接反量化再量化的效率损失。

**用数据推演位移方法（M1）**：

```
原始 s = 0.0472
ŝ = 2^⌈log₂(0.0472)⌉ = 2^⌈-4.4⌉ = 2^(-4) = 0.0625

0.5 → round(0.5/0.0625)=8  → 反量化 0.5    ✅ 更精确了！
0.1 → round(0.1/0.0625)=2  → 反量化 0.125  ❌ 误差 0.025
0.3 → round(0.3/0.0625)=5  → 反量化 0.3125 ❌ 误差 0.0125
6.0 → round(6.0/0.0625)=96 → 反量化 6.0    ✅

为什么用 2 的幂？因为位移操作比浮点乘法快得多。
```

> 🔍 **两种位移方法**
> - **(M1)**：映射到由 2 的幂表示的最接近的值 `Ŝ = 2^⌈log₂(S)⌉`
> - **(M2)**：收集 scales 形成向量，取组内最大值 S_max，将 S_max/S_i 调整为 2 的幂表示 `Ŝ_i = S_max / 2^⌈log₂(S_max/S_i)⌉`，近似更优
>
> **补充**：S=2ⁿ 中，当 n 为负数时表示小数（如 2⁻⁴=0.0625），当 n 不为负数时 S 可以表示整数。**无论用 (M1) 还是 (M2)，把 scale 限制为 2 的幂都可以简化计算，特别是在基于二进制逻辑操作的数字系统中**（位移代替乘除法），这是同时优化计算效率和保持模型性能的关键要素。

---

## 2.10 FP6-LLM

**特点**：W6A16

FP6 量化的一个挑战是缺乏针对这种不规则位宽的高效 GPU Kernel 设计。FP6-LLM 中设计并实现了 **TC-FPx**，第一个具有 Tensor Core 支持的用于 FP6 和各种量化位宽（6位、5位、3位等）的浮点权重的 GPU 系统设计方案，缓解了 LLM 推理期间的"内存墙"问题。

### 关键创新

1. **运行前比特层级的数据排布转换**：解决权重具有不规则位宽时不友好的内存访问挑战。
2. **运行时高效的 SIMT 计算**：最小化权重反量化的运行时开销。
3. **全栈的高效流水线设计**：SIMT Core 计算、Tensor Core 计算和 GPU 内存访问高效调度。

### 性能与限制

- 在 NVIDIA A100 上，FP6 kernel 处理速度比 FP16 cuBLAS 基准提高 2.1 倍（decoder 矩阵形状狭长，访存瓶颈）。
- FP6 量化使 LLaMA-70B 模型能在单个 A100 GPU 上运行。
- 批处理大小小于 32 时，性能比 FP16 基准高出 1.69 到 2.65 倍。
- **限制**：当 GEMM 瓶颈变为 Tensor Core 计算（大批量或充足显存）时，仅权重量化 kernel 可能无法保持性能优势；目前仅支持非 MoE 结构，仅与 FP16 输入模型兼容。

---

## 2.11 KIVI

**特点**：在线量化、对 Token 的 KV 缓存进行分组，在组内对 K 缓存逐通道量化，对 V 缓存逐 Token 量化，KV2

KIVI 开发了一种 2 位非对称 KV 缓存量化方法。通过实验证明对 **Key 缓存按通道进行量化，对 Value 缓存按 Token 进行量化**能够有效保持模型的精度。

### 为什么 Key 逐通道、Value 逐 Token？

- Key 的异常值按通道聚集（同一通道中的大幅度值组合在一起），逐通道量化可以最小化量化误差。
- Value 的异常值按 Token 聚集，逐 Token 量化更合适。
- 从实现角度：逐 Token 的 Value 量化可以与流式场景很好地结合（新量化的张量按 Token 维度直接附加）；逐通道的 Key 量化跨越不同 Token，不能直接在流式场景中实现。

### KIVI 的分组设计

将 Token 的 KV 缓存分成两部分：**分组部分**和**余留部分**。

- **分组部分**：每 G 个 Token 的 KV 缓存分为一组。在组内分别进行逐 Token 的 Value 缓存量化和逐通道的 Key 缓存量化。
- **余留部分**：保留全精度不进行量化（凑不齐一组的剩余 Token）。

### 为什么需要余留部分？

Key 按通道量化需要"一整列"的数据才能算 scale。新 token 到达时只有 `K_new[1, head_dim]`，无法立刻知道最终 range。KIVI 的办法：先放余留部分（FP16），凑够 G 个再一起量化，最新一小段保持全精度。**quantized groups + full precision residual window**。

### 用数据推演 KIVI（假设 group G=2）

假设 KV 缓存逐步累积：

```
Key 缓存（逐步累积）：
         c0   c1   c2   c3
step1: [0.5, 0.2, 0.8, 0.3]
step2: [0.4, 0.6, 0.2, 0.7]
step3: [0.8, 5.0, 0.3, 0.1]   ← step3 的 c1 突然出现离群值 5.0

Value 缓存（逐步累积）：
         c0   c1   c2   c3
step1: [0.3, 0.7, 0.1, 0.5]
step2: [0.6, 0.2, 0.4, 0.8]
step3: [0.1, 0.4, 0.9, 0.2]
```

**流程**：生成第 1 个 token 时余留部分 FP16 不量化 → 生成第 2 个 token 凑够 G=2，组内量化：

```
Key 按通道量化（per-channel，INT2 非对称，范围用 max 归一化）：
  c0 列: [0.5, 0.4] → s_c0 = 0.8/1 = 0.8
  c1 列: [0.2, 0.6] → s_c1 = 0.6/1 = 0.6
  c2 列: [0.8, 0.2] → s_c2 = 0.8/1 = 0.8
  c3 列: [0.3, 0.7] → s_c3 = 0.7/1 = 0.7

Value 按 token 量化（per-token）：
  step1: [0.3, 0.7, 0.1, 0.5] → s_step1 = 0.7/1 = 0.7
  step2: [0.6, 0.2, 0.4, 0.8] → s_step2 = 0.8/1 = 0.8
```

生成第 3 个 token（c1 有离群值 5.0）时，还在余留部分保持 FP16，精度无损；等凑够一组再量化。

> 🔍 **学习追问：KIVI 是不是要记住每个 group 每个 V 的每个 token 的 s 和 z？K 逐通道量化也和 V 差不多？**
>
> **是的。** 量化后所有 scale 和 zero 都要存下来用于反量化：
> - **V 量化**：逐 Token 非对称量化，每个 Token 的 V 向量需要存一组 (s, z)。group=G 时，一个组内 G 个 Token 就存 G 组。
> - **K 量化**：逐通道非对称量化，每个通道需要存一组 (s, z)。hidden_dim=D 时，存 D 组。
> - K 和 V 的量化方式一样（都是 2bit 非对称），只是"按什么分组"不同：V 按 Token 分，K 按通道分。

> 🔍 **存储开销问题（重要）**
>
> 位宽越低，scale/zero 的元数据占比越大：
> ```
> 对于 G=2，d=4 的一个组：
> 不量化（FP16）：K/V 各 2×4×16bit，总计 256bit
> KIVI（INT2）：K/V 数据 32bit + K_scale/zero 128bit + V_scale/zero 64bit = 224bit
> → 仅节省 12.5%（元数据开销大）
>
> G=128 时：K_scale 4×16bit = 64bit 由 128 个 token 分摊，几乎可忽略
> ```
> 这就是为什么 KIVI 要做**分组量化**（G 越大，元数据被分摊得越薄）。KVQuant 对 scale 做进一步量化（FP16→FP8）也是为减少元数据开销。

> 🔍 **KIVI 一句话**
>
> KIVI 基于 KV 分布差异采用 2bit asymmetric quantization：K 按 channel、V 按 token；为解决自回归时 K per-channel 无法对未来 token 立即确定 range 的问题，维护未量化 residual window，凑够 group 后再批量量化。

---

**博主原文配图**（源自知乎原文）：

![原文图 12](https://picx.zhimg.com/v2-b7f73623f5424289583c400f8853e48c_r.jpg?source=2c26e567)


---

### 📌 原始 Q&A：KIVI（历史第 72-74 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 72 轮）


---
我们来讨论下这部分 KIVI

##### A（历史第 73 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！用我们的 4×4 数据完整推演 **KIVI**。

---

###### 设定回顾

为了推演 KIVI，我们需要模拟 **自回归生成** 的过程。假设模型正在生成一个序列，每次生成一个 token：

```
已生成的 token 序列（模拟）：
  step1: 生成 token1 → 计算 K₁[4], V₁[4]
  step2: 生成 token2 → 计算 K₂[4], V₂[4]
  step3: 生成 token3 → 计算 K₃[4], V₃[4]
  
每一步生成的 KV 都追加到 KV Cache 中
```

假设生成的 KV 数据如下：

```
Key 缓存（逐步累积）：
         c0   c1   c2   c3
step1: [0.5, 0.2, 0.8, 0.3]
step2: [0.4, 0.6, 0.2, 0.7]
step3: [0.8, 5.0, 0.3, 0.1]   ← step3 的 c1 突然出现离群值 5.0

Value 缓存（逐步累积）：
         c0   c1   c2   c3
step1: [0.3, 0.7, 0.1, 0.5]
step2: [0.6, 0.2, 0.4, 0.8]
step3: [0.1, 0.4, 0.9, 0.2]
```

---

###### 一、KIVI 的核心思想

答主原文：

> KIVI 开发了一种 **2 位非对称 KV 缓存量化** 方法。通过实验证明 **对 Key 缓存按通道进行量化，对 Value 缓存按 Token 进行量化** 能够有效保持模型的精度。

**KIVI 就是量化 KV Cache 的，不是量化权重或激活。而且是 KV2（2bit）！**

---

###### 二、为什么需要这个方案？

KV Cache 的问题是 **逐 token 到达的**：

```
生成第 1 个 token 时：
  KV Cache = [K₁, V₁]          只有 1 个 token

生成第 2 个 token 时：
  KV Cache = [K₁, V₁, K₂, V₂]  第 2 个来了

生成第 3 个 token 时：
  KV Cache = [K₁, V₁, K₂, V₂, K₃, V₃]  第 3 个来了
```

**你永远不知道下一个 token 的 KV 会是多少。** 不能像量化权重那样"先扫一遍全部数据再定 scale"。

而且 **Key 和 Value 的分布不同**：

```
Key 的 c1 列：step1=0.2, step2=0.6, step3=5.0
  → 某些通道跨 token 变化很大，按通道量化更合适

Value 的各行：每行有自己的数值范围
  → 按 token 量化更自然
```

---

###### 三、KIVI 的解决方案

KIVI 把 Token 的 KV 缓存分成两部分：

```
KV Cache 内部：
┌──────────────────────────────────────────────┐
│  分组部分（每 G 个 token 一组）               │
│  组内：Key 逐通道量化，Value 逐 Token 量化    │
├──────────────────────────────────────────────┤
│  余留部分（不满 G 个 token）                  │
│  保持全精度 FP16，不量化                     │
└──────────────────────────────────────────────┘
```

---

###### 四、用数据一步步推演（假设 group G=2）

###### 第 1 步：生成第一个 token

```
K₁ = [0.5, 0.2, 0.8, 0.3]
V₁ = [0.3, 0.7, 0.1, 0.5]

此时 KV Cache 只有 1 个 token，不到 G=2：
  余留部分 = [K₁, V₁]（FP16，不量化）
```

###### 第 2 步：生成第二个 token

```
K₂ = [0.4, 0.6, 0.2, 0.7]
V₂ = [0.6, 0.2, 0.4, 0.8]

此时 KV Cache 有 2 个 token，达到 G=2，可以组成一个组了：
  分组 = [K₁, V₁, K₂, V₂]
```

**组内量化：**

**Key 按通道量化（per-channel，2bit）**：

```
Key 的 2 个 token，4 个通道（列）：

c0 列: [0.5, 0.4]  → max=0.5, min=0.4
c1 列: [0.2, 0.6]  → max=0.6, min=0.2
c2 列: [0.8, 0.2]  → max=0.8, min=0.2
c3 列: [0.3, 0.7]  → max=0.7, min=0.3

INT2 非对称量化（4 个格子：0, 1, 2, 3）：

c0: s=(0.5-0.4)/3=0.0333, z=0（假设）
  0.5→round((0.5-0.4)/0.0333)=round(3.0)=3→反量化 3×0.0333+0.4=0.5 ✅
  0.4→round((0.4-0.4)/0.0333)=0→反量化 0.4 ✅

c1: s=(0.6-0.2)/3=0.1333
  0.2→0→0.2 ✅
  0.6→3→0.6 ✅

c2: s=(0.8-0.2)/3=0.2
  0.8→3→0.8 ✅
  0.2→0→0.2 ✅

c3: s=(0.7-0.3)/3=0.1333
  0.3→0→0.3 ✅
  0.7→3→0.7 ✅

2bit 量化下精度很好，因为组内两个 token 的 value 范围接近。
```

**Value 按 token 量化（per-token，2bit）**：

```
V₁ = [0.3, 0.7, 0.1, 0.5] → max=0.7, min=0.1, s=(0.7-0.1)/3=0.2
  0.3→1→0.3, 0.7→3→0.7, 0.1→0→0.1, 0.5→2→0.5 ✅

V₂ = [0.6, 0.2, 0.4, 0.8] → max=0.8, min=0.2, s=(0.8-0.2)/3=0.2
  0.6→2→0.6, 0.2→0→0.2, 0.4→1→0.4, 0.8→3→0.8 ✅
```

**量化后存起来，同时清除余留部分。**

###### 第 3 步：生成第三个 token

```
K₃ = [0.8, 5.0, 0.3, 0.1]  ← c1 出现离群值 5.0！
V₃ = [0.1, 0.4, 0.9, 0.2]

此时 KV Cache：
  ┌──────────────────────────────────────────────┐
  │  已量化分组：组1（step1, step2）             │
  │    K_int2: [c0_s, c1_s, c2_s, c3_s] + index │
  │    V_int2: [V₁_int2, V₂_int2]               │
  ├──────────────────────────────────────────────┤
  │  余留部分（FP16）：[K₃, V₃]  ← 不到 G=2     │
  └──────────────────────────────────────────────┘
```

**K₃ 的 c1=5.0 有问题吗？** 没问题！因为它目前还在余留部分，保持 FP16 不量化，精度无损。

###### 第 4 步：生成第四个 token

```
K₄ = [0.3, 0.5, 0.7, 0.2]
V₄ = [0.5, 0.1, 0.3, 0.7]

此时余留部分达到 G=2：[K₃, V₃, K₄, V₄]
形成一个新组，开始组内量化。
```

**Key 按通道量化（这次 c1 列有离群值 5.0）：**

```
c1 列: [5.0, 0.5]  ← 一个 5.0，一个 0.5，范围跨度很大！
  INT2: s=(5.0-0.5)/3=1.5
  5.0→round((5.0-0.5)/1.5)=round(3.0)=3→反量化 3×1.5+0.5=5.0 ✅
  0.5→0→0.5 ✅

2bit 居然也能覆盖！因为非对称量化（z≠0）能处理偏移很大的数据。
但如果用对称量化：
  s=5.0/3=1.667
  0.5→round(0.5/1.667)=0→0 ❌（误差 -0.5）
  
所以 KIVI 用非对称量化来处理这种分布。
```

---

###### 五、为什么 Key 按通道、Value 按 token？

###### Key 按通道

**Key 的某些通道在不同 token 间有"全局"特征**：

```
c1 列：[step1=0.2, step2=0.6, step3=5.0, step4=0.5]
  → c1 通道本身"就是这个范围"，每个 token 的 c1 值都是这个量级
  → 按通道量化，用同一个 scale 覆盖整列

如果按 token 量化 Key：
  token3: [0.8, 5.0, 0.3, 0.1]
    s=5.0/3=1.667（2bit 对称）
    0.8→0→0 ❌ 小值全死了
```

###### Value 按 token

**Value 没有明显的通道性离群值，但不同 token 的 Value 范围不同**：

```
V₁: [0.3, 0.7, 0.1, 0.5] → range=[0.1, 0.7]
V₂: [0.6, 0.2, 0.4, 0.8] → range=[0.2, 0.8]
V₃: [0.1, 0.4, 0.9, 0.2] → range=[0.1, 0.9]

没有哪个列特别大或特别小，所以按 token 量化（每行自己的范围）。
```

---

###### 六、余留部分的作用

答主原文：

> 分组部分将每 G 个 Token 的 KV 缓存分为一组。在组内分别进行逐Token的Value缓存量化和逐通道的Key缓存量化。而**余留部分则保留全精度不进行量化**。

**为什么需要余留部分？**

```
Key 按通道量化需要"一整列"的数据才能算 scale。

如果不用余留部分：
  step1 来了 K₁ → 只有 1 个 token，怎么算 c0 列的 scale？
  只能等 step2 来了再一起算 → 那 step1 的数据放哪？先量化？精度会差。

KIVI 的方案：
  先放余留部分（FP16）→ 凑够 G 个 → 一起量化 → 移到已量化分组
```

---

###### 七、KIVI 的整体流程

```
初始状态：KV Cache 为空

每个生成步骤：
  1. 生成新 token 的 K, V
  2. 追加到余留部分（FP16）
  3. 如果余留部分的 token 数 == G：
     a. 把余留部分的 G 个 token 拿出来
     b. Key 按通道 2bit 量化，存 int2 + scale + zero
     c. Value 按 token 2bit 量化，存 int2 + scale + zero
     d. 清空余留部分
  4. 否则：继续留在余留部分，保持 FP16

推理时（Attention 计算）：
  1. 已量化分组：反量化 → FP16
  2. 余留部分：直接 FP16（本来就没量化）
  3. 合并两部分 → 完整的 KV Cache（FP16）
  4. 正常做 Attention
```

---

###### 八、KIVI 和 KVQuant 的对比

答主原文也提到了 KVQuant，它们的区别：

| | KIVI | KVQuant |
|:--|:----|:--------|
| 量化位宽 | KV2 | KV2 / KV3 |
| Key 量化方式 | per-channel（组内） | per-channel（离线校准） |
| Value 量化方式 | per-token | per-token |
| 是否需要分组 | 需要（G 个一组） | 不需要（用离线校准集） |
| 是否需要余留部分 | 需要 | 不需要 |
| 是否需要校准集 | 不需要（在线量化） | 需要（离线量化） |

---

###### 九、一句话总结

> **KIVI 是 KV Cache 的 2bit 在线量化方案。把生成的 token 按 G 个一组分组，组内 Key 按通道量化、Value 按 token 量化，凑不齐一组的先放余留部分保持 FP16。不需要校准集，推理时逐 token 量化。**

##### Q（历史第 74 轮）

所以 KIVI的量化其实就是这样。先设定参数 group的参数。 然后等凑齐group个的KV cache的话就分别对该组的K进行量化。对该组的V进行量化。然后V量化的话就直接是一个token自己的那一行数据进行量化。比如用2bit的非对称量化。；；那这样是不是要记住该group该V的该token的s和z啊？ 然后对K进行量化的话就是逐列进行量化了。也和V的量化差不多？

##### A（历史第 75 轮，正文原样保留；仅将内部标题降级以保持目录层级）

你的理解 **基本完全正确！** 我帮你精确确认几个细节。

---

###### 你的理解 ✅

| 你说的 | 对不对 |
|:------|:------|
| 先设定 group 参数 G | ✅ 对的 |
| 凑齐 G 个 token 后一起量化 | ✅ 对的 |
| 该组的 K 做量化，该组的 V 做量化 | ✅ 对的 |
| V 按行（每个 token 自己的数据）量化 | ✅ 对的 |
| K 按列（逐通道）量化 | ✅ 对的 |
| 都要记住各自的 (s, z) | ✅ **对的！** |

---

###### 需要存多少 (s, z)？

用我们的数据，假设 G=2，d=4，2bit 非对称量化：

###### V 的 (s, z)：每行存一套

```
V 组内 2 个 token：
         c0   c1   c2   c3
step1: [0.3, 0.7, 0.1, 0.5]  → 算得 (s₁, z₁)
step2: [0.6, 0.2, 0.4, 0.8]  → 算得 (s₂, z₂)

存储：
  V_int2: [索引(step1的4个2bit), 索引(step2的4个2bit)]
  V_scale: [s₁, s₂]   ← 每个 token 一个 scale
  V_zero:  [z₁, z₂]   ← 每个 token 一个 zero
```

**V 存了 2 套 (s, z)，对应 2 个 token。**

###### K 的 (s, z)：每列存一套

```
K 组内 2 个 token：
         c0    c1    c2    c3
step1: [0.5,  0.2,  0.8,  0.3]
step2: [0.4,  0.6,  0.2,  0.7]

c0 列: [0.5, 0.4] → 算得 (s_c0, z_c0)
c1 列: [0.2, 0.6] → 算得 (s_c1, z_c1)
c2 列: [0.8, 0.2] → 算得 (s_c2, z_c2)
c3 列: [0.3, 0.7] → 算得 (s_c3, z_c3)

存储：
  K_int2: [每个位置存 2bit 索引]
  K_scale: [s_c0, s_c1, s_c2, s_c3]  ← 每列一个 scale
  K_zero:  [z_c0, z_c1, z_c2, z_c3]  ← 每列一个 zero
```

**K 存了 4 套 (s, z)，对应 4 个通道（列）。**

---

###### K 和 V 量化方式一样，只是 "按什么分组" 不同

**V 的量化（per-token，按行）**：

```
一行（一个 token）的所有列共用一套 (s, z)
V_step1: [0.3, 0.7, 0.1, 0.5] → 算一套 (s₁, z₁)
  max=0.7, min=0.1
  s = (0.7-0.1)/3 = 0.2
  0.3→1, 0.7→3, 0.1→0, 0.5→2
```

**K 的量化（per-channel，按列）**：

```
一列（一个通道）的所有行共用一套 (s, z)
K_c0: [0.5, 0.4] → 算一套 (s_c0, z_c0)
  max=0.5, min=0.4
  s = (0.5-0.4)/3 = 0.0333
  0.5→3, 0.4→0
```

**本质都是非对称 2bit 量化，只是"谁和谁共用一套 (s, z)"不同。**

---

###### 存储开销

对于 G=2，d=4 的一个组：

```
不量化（FP16）：
  K: 2×4×16bit = 128bit
  V: 2×4×16bit = 128bit
  总计：256bit

KIVI 量化（INT2）：
  K_int2: 2×4×2bit = 16bit
  K_scale: 4×16bit = 64bit（每列一个 FP16 scale）
  K_zero:  4×16bit = 64bit（每列一个 FP16 zero）
  V_int2: 2×4×2bit = 16bit
  V_scale: 2×16bit = 32bit（每 token 一个 FP16 scale）
  V_zero:  2×16bit = 32bit（每 token 一个 FP16 zero）
  总计：16+64+64+16+32+32 = 224bit

224bit vs 256bit → 节省了 12.5%
```

等等，**2bit 量化的 scale/zero 开销都快赶上数据本身了**？

是的，位宽越低，scale/zero 的元数据占比越大。这就是为什么 KIVI 要做 **分组量化**（G 越大，元数据被分摊得越薄）。

```
G=2 时：元数据占比很高
G=128 时：
  K: 128×4×2bit = 1024bit
  K_scale: 4×16bit = 64bit ← 128 个 token 分摊，几乎可以忽略
```

另外答主还提到 KVQuant 对 scale 做了**进一步量化**（比如把 FP16 的 scale 再量化为 FP8），也是为减少元数据开销。

---

###### 一句话总结你的理解

> **KIVI 每 G 个 token 一组，组内 V 按行（每个 token 一套 s,z）、K 按列（每个通道一套 s,z）分别做 2bit 非对称量化。所有 scale 和 zero 都要存下来用于反量化。**

## 2.12 IntactKV

**特点**：保持关键词元的 KV 缓存无损

IntactKV 先使用全精度模型生成关键词元的无损 KV 并将其缓存下来，量化模型在推理时就能直接使用无损的关键词元表征，从而有效提升量化模型精度。

通过利用 LLM 推理时的**公共前缀**防止给量化模型的推理带来任何额外开销。即 IntactKV 只包含公共前缀，那么全精度模型在生成 IntactKV 后就可以被释放。

此外，缓存的 IntactKV 还可以作为模型的额外参数做校准来进一步弥补量化误差。比如：在大小为 128 的校准集上做轻量化训练，可学习参数仅为缓存下来的 IntactKV，损失函数采用量化模型和全精度模型输出的 MSE 损失。

### 用数据看

```
生成序列："我 爱 吃 苹果"
第一个 token "我" 的 KV 缓存保留 FP16 不量化：
  K_<s> = [0.5, 0.2, 0.8, 0.3]  ← FP16
  V_<s> = [0.3, 0.7, 0.1, 0.5]  ← FP16
后面的 token 正常量化：
  K_爱 = INT4, K_吃 = INT4, K_苹果 = INT4
```

### 核心思想

用全精度模型提前生成公共前缀（system prompt、few-shot 例子等固定文本）的 KV Cache（FP16 无损），量化模型推理时直接用这个无损 KV，后面的 token 正常量化。

**流程**：
1. 用全精度模型跑一次公共前缀，得到无损 KV，缓存下来；全精度模型随后释放。
2. 加载量化模型，把 IntactKV 加载到 KV Cache 初始位置。
3. 后续 token 的 KV 由量化模型正常量化生成。
4. Attention 时所有 K/V 反量化回 FP16 计算。

### 解决了什么问题（Attention Sink）

由于 Attention Sink 现象，模型对第一个 Token 中的量化误差异常敏感——所有 token 都会关注第一个 token，如果它的 KV 有量化误差，误差会被所有后续 token 的 attention 放大，污染整个序列的注意力分布。IntactKV 让第一个 token（公共前缀）的 KV 保持 FP16 无损，彻底避免这个问题。

> ⚠️ **易错校正：IntactKV ≠ "永远只有第一个 token 保 FP16"**
>
> IntactKV 的核心概念是 **pivot tokens / pivot KV**（保护极少数关键 token 的 KV，使其不受量化误差污染），可以与公共前缀结合，不必机械等同于"永远只有第一个 token"。

### IntactKV vs KVQuant 的 Attention Sink 策略对比

| | IntactKV | KVQuant |
|:--|:--------|:--------|
| 做法 | 全精度模型先生成，缓存无损 KV | 第一个 token 保留 FP16 |
| 适用场景 | 有公共前缀（system prompt, few-shot） | 通用场景 |
| 额外成本 | 需要额外跑一次全精度模型 | 不需要 |
| 是否可微调 | ✅ 可以做校准 | 固定的 |

---

### 📌 原始 Q&A：IntactKV（历史第 76 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 76 轮）

那么我们现在讨论这部分：IntactKV

##### A（历史第 77 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！用我们的数据来推演 **IntactKV**。

---

###### 设定

继续用 KV Cache 的场景。假设模型生成序列：

```
已生成的序列："我 爱 吃 苹果"
对应的 KV Cache（逐步累积）：
```

**K 缓存（4×4）：**
```
         c0   c1   c2   c3
我(step1): [0.5, 0.2, 0.8, 0.3]
爱(step2): [0.4, 0.6, 0.2, 0.7]
吃(step3): [0.8, 5.0, 0.3, 0.1]
苹果(step4): [0.3, 0.5, 0.7, 0.2]
```

**V 缓存（4×4）：**
```
         c0   c1   c2   c3
我(step1): [0.3, 0.7, 0.1, 0.5]
爱(step2): [0.6, 0.2, 0.4, 0.8]
吃(step3): [0.1, 0.4, 0.9, 0.2]
苹果(step4): [0.5, 0.1, 0.3, 0.7]
```

---

###### 一、IntactKV 的核心思想

答主原文：

> **IntactKV 先使用全精度模型生成关键词元的无损 KV 并将其缓存下来，量化模型在推理时就能直接使用无损的关键词元表征，从而有效提升量化模型精度。**

**核心：保持前几个 token（通常是公共前缀）的 KV Cache 不量化，保持全精度。后面的 token 正常量化。**

---

###### 二、什么是"关键词元"？

在 LLM 推理中，很多场景有 **公共前缀**：

```
对话场景：
  System Prompt: "你是一个有用的助手，请用中文回答用户的问题。"
  ↓
  每次用户提问，这个 System Prompt 的 KV Cache 都是一样的

Few-shot 场景：
  Prompt: "以下是一些例子：\n例子1: ... \n例子2: ... \n请回答："
  ↓
  每次用户提问，这些例子的 KV Cache 都是一样的
```

IntactKV 就是把这些公共前缀的 KV Cache **用全精度模型一次性算好，保持无损**，然后量化模型推理时直接拿来用。

---

###### 三、用数据推演

假设 sequence 的首个 token "我" 是公共前缀（比如 `<s>` 或 "请回答："）。

###### 第 1 步：用全精度模型生成 "我" 的 KV

```
用全精度（FP16）模型跑一次 "我"：
  K_我 = [0.5, 0.2, 0.8, 0.3]  ← FP16，无损
  V_我 = [0.3, 0.7, 0.1, 0.5]  ← FP16，无损

把这些存下来。
全精度模型就可以释放了（不用再加载了）。
```

###### 第 2 步：加载量化模型开始推理

```
加载 W4A16 量化模型（比如 GPTQ 量化过的）。

把 IntactKV（K_我, V_我）加载到 KV Cache 中：
  KV Cache 初始状态：
    K = [0.5, 0.2, 0.8, 0.3]  ← FP16，无损
    V = [0.3, 0.7, 0.1, 0.5]  ← FP16，无损
```

###### 第 3 步：推理后续 token

开始生成 "爱"：

```
输入 = "我"的最后一个 hidden state（从全精度模型得到的）
量化模型前向 → 生成 K_爱, V_爱
  K_爱 = [0.4, 0.6, 0.2, 0.7]  ← 用 INT4 量化存
  V_爱 = [0.6, 0.2, 0.4, 0.8]  ← 用 INT4 量化存
```

继续生成 "吃"：

```
输入 = "爱"的 hidden state
量化模型前向 → 生成 K_吃, V_吃
  K_吃 = [0.8, 5.0, 0.3, 0.1]  ← INT4
  V_吃 = [0.1, 0.4, 0.9, 0.2]  ← INT4
```

继续生成 "苹果"：

```
输入 = "吃"的 hidden state
量化模型前向 → 生成 K_苹果, V_苹果
  K_苹果 = [0.3, 0.5, 0.7, 0.2]  ← INT4
  V_苹果 = [0.5, 0.1, 0.3, 0.7]  ← INT4
```

###### 第 4 步：Attention 计算

```
最终的 KV Cache：
  K = [FP16无损, INT4, INT4, INT4]
  V = [FP16无损, INT4, INT4, INT4]

做 Attention 时：
  1. 所有 K/V 反量化回 FP16（包括 INT4 的）
  2. 和正常的 Attention 计算一样做 Q·Kᵀ · V
```

---

###### 四、IntactKV 也能做校准

答主原文：

> 此外，缓存的 IntactKV 还可以作为模型的额外参数做校准来进一步弥补量化误差。

```
在 128 条校准数据上做轻量化训练：
  可学习参数：IntactKV（就是 K_我, V_我 那部分）
  损失函数：量化模型输出和全精度模型输出的 MSE
```

**也就是说 IntactKV 不仅仅是"保持无损"，还可以被微调，让量化模型在第一批 token 上的输出更接近全精度模型。**

---

###### 五、IntactKV 解决了什么问题？

答主原文在 KVQuant 里提到过 **Attention Sink 现象**：

> 由于 Attention Sink 现象，模型对第一个 Token 中的量化误差异常敏感。

第一个 token（通常是 `<s>` 或 `BOS`）是所有 token 的 attention 焦点，即使它的值本身不重要，所有其他 token 都会关注它。

如果第一个 token 的 KV 被量化了，误差会被 **所有后续 token 的 attention 放大**。

```
Attention Sink 现象：
  所有 token 都会关注第一个 token（即使它没什么信息量）
  ↓
  如果第一个 token 的 K 有量化误差 → 所有 Q·Kᵀ 都有误差
  ↓
  整个序列的注意力分布都被污染了
```

IntactKV 的做法是：**第一个 token 的 KV 保持 FP16 无损，彻底避免这个问题。**

---

###### 六、IntactKV 和 KVQuant 的 Attention Sink 策略对比

| | IntactKV | KVQuant |
|:--|:--------|:--------|
| 做法 | 全精度模型先生成，缓存无损 KV | 第一个 token 保留 FP16 |
| 适用场景 | 有公共前缀（system prompt, few-shot） | 通用场景 |
| 额外成本 | 需要额外跑一次全精度模型 | 不需要 |
| 是否可微调 | ✅ 可以做校准 | 固定的 |

---

###### 七、答主原文的关键句

> 通过利用 LLM 推理时的公共前缀防止给量化模型的推理带来任何额外开销。即 IntactKV 只包含公共前缀，那么全精度模型在生成 IntactKV 后就可以被释放。

> 缓存的 IntactKV 还可以作为模型的额外参数做校准来进一步弥补量化误差。

---

###### 八、一句话总结

> **IntactKV 用全精度模型提前生成公共前缀的 KV Cache（FP16 无损），量化模型推理时直接用这个无损 KV，后面的 token 正常量化。这样既避免了 Attention Sink 的误差放大，又不需要在推理时额外跑全精度模型。**

## 2.13 KVQuant

**特点**：离线量化、基于校准集量化无需进行分组

KVQuant 通过结合几种新颖的方法来缓解 KV 缓存量化的精度损失。

### 六个优化技巧

1. **逐通道（Per-Channel）对 Key 进行量化**：通过调整 Key 激活的量化维度以更好地匹配分布。利用离线校准集准确执行按通道量化，不需要分组。
2. **在 RoPE 之前对 Key 进行量化**：在旋转位置嵌入之前量化 Key 激活，以减轻其对量化的影响。开发了一个融合 Kernel，以在反量化后高效应用 RoPE。
3. **非均匀 KV 缓存量化**：通过导出每层敏感度加权的非均匀数据类型，以更好地表示分布。
4. **按向量（Per-Vector）密集和稀疏量化**：分别隔离每个向量的异常值，以最大限度地减少量化范围中的偏差。
5. **Attention Sink 感知量化**：由于 Attention Sink 现象，模型对第一个 Token 中的量化误差异常敏感。通过仅保留第一个 Token 为 FP16 来确保模型的精度。
6. **离线计算缩放因子**：逐通道量化中，离线（使用校准数据）计算统计数据，避免在线更新缩放因子。逐 Token 的 Value 量化则每个传入 Token 在线计算缩放因子和异常值阈值，通过卸载到 CPU 来高效计算。

> 🔍 **第 6 技巧的 kernel 细节**：由于存在异常的 Value tokens，离线校准缩放因子对 Value 是有挑战的，因此 Value 采用**在线**量化——通过**利用定制的量化函数**实现来压缩激活，可以**在线执行逐 Token 的 Value 量化**而不会影响性能；Key 由于可以用离线校准集固定 per-channel scale，避免了 KV 缓存添加新 Token 时在线更新缩放因子的需要。

### 用数据看关键技巧

```
① RoPE 之前量化 Key：
  量化前：K = [0.5, 0.2, 0.8, 0.3]  ← 范围小，容易量化
  应用 RoPE：K_rotated = 旋转后的值  ← 范围变大，更难量化
  → 所以 KVQuant 在 RoPE 之前量化 Key

② Attention Sink 感知量化：
  第一个 token 保留 FP16：K_<s> = [0.5, 0.2, 0.8, 0.3]（FP16 不量化）
  后面的 token 正常量化：K_step2 = INT4, K_step3 = INT4, ...

③ Per-vector 密集+稀疏量化：
  单行内异常值单独拎出来用稀疏表示，避免拉大 range
```

### 6 个技巧分别解决什么问题

| 技巧 | 解决的问题 |
|:----|:----------|
| 1. Per-channel 量化 Key | Key 的通道性离群值（如 c1 列有 5.0） |
| 2. RoPE 前量化 Key | RoPE 改变分布（旋转后值范围变大、出现负数）加大量化难度 |
| 3. 非均匀量化 | 不同层分布不同（有的偏正态、有的偏均匀） |
| 4. Per-vector 密集+稀疏 | 单行内的异常值单独拎出来，避免拉大 range |
| 5. Attention Sink 感知 | 第一个 token 保持 FP16，避免误差被放大 |
| 6. 离线 scale | 避免在线更新需要重新量化已有数据 |

### KVQuant vs KIVI

| 维度 | KVQuant | KIVI |
|---|---|---|
| 量化时机 | 离线（校准集）+ 在线（Value） | 在线（生成时） |
| Key 量化 | 逐通道，无需分组 | 逐通道，需要分组 |
| Value 量化 | 逐 Token，在线计算 | 逐 Token，分组内量化 |
| Attention Sink | ✅ 第一个 Token 保留 FP16 | ❌ 无特殊处理 |
| RoPE 前量化 | ✅ 融合 Kernel | ❌ 无 |
| 是否需要校准集 | ✅ 需要 | ❌ 不需要 |

**KIVI 和 KVQuant 的核心区别**：KIVI 在线做（不依赖校准集，用 residual + group 解决 K per-channel），KVQuant 离线做（用 calibration 提前估计固定 K channel scale，因此不需要等待未来 token 更新 K scale）。

---

**博主原文配图**（源自知乎原文）：

![原文图 13](https://pica.zhimg.com/v2-678830ea58f41d98e86a803c38a98b3b_r.jpg?source=2c26e567)


![原文图 14](https://picx.zhimg.com/v2-4998abb3fc753bfe24997f190dd9e027_r.jpg?source=2c26e567)


---

### 📌 原始 Q&A：KVQuant（历史第 78 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 78 轮）

接下来我们讨论这部分：KVQuant

##### A（历史第 79 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！用我们的 4×4 数据推演 **KVQuant**。

---

###### 设定

继续用 KV Cache 的场景，假设生成序列：

```
K 缓存（4×4）：
         c0   c1   c2   c3
step1: [0.5, 0.2, 0.8, 0.3]
step2: [0.4, 0.6, 0.2, 0.7]
step3: [0.8, 5.0, 0.3, 0.1]
step4: [0.3, 0.5, 0.7, 0.2]

V 缓存（4×4）：
         c0   c1   c2   c3
step1: [0.3, 0.7, 0.1, 0.5]
step2: [0.6, 0.2, 0.4, 0.8]
step3: [0.1, 0.4, 0.9, 0.2]
step4: [0.5, 0.1, 0.3, 0.7]
```

---

###### 一、KVQuant 的核心思想

答主原文：

> **KVQuant 通过结合几种新颖的方法来缓解 KV 缓存量化的精度损失。**

KVQuant 不是靠"一个技巧"打天下，而是把 **多个优化技巧组合在一起**，逐个解决 KV 量化中的问题。

---

###### 二、KVQuant 的 6 个技巧

答主原文列出了 6 个技巧：

1. **逐通道（Per-Channel）对 Key 进行量化**
2. **在 RoPE 之前对 Key 进行量化**
3. **非均匀 KV 缓存量化**
4. **按向量（Per-Vector）密集和稀疏量化**
5. **Attention Sink 感知量化**
6. **离线计算缩放因子**

---

###### 三、技巧 1：逐通道对 Key 量化

KIVI 也用了 per-channel 量化 Key，但 KVQuant 的做法不同。

**KIVI 的做法**：需要分组（G 个 token 一组），组内 per-channel 量化，还要有余留部分。

**KVQuant 的做法**：**不需要分组**，直接用离线校准集算好每列的 scale，推理时直接用。

```
用校准集跑一遍，收集 Key 的统计信息：
  c0 列: [0.5, 0.4, 0.8, 0.3] → 离线算好 s_c0, z_c0
  c1 列: [0.2, 0.6, 5.0, 0.5] → 离线算好 s_c1, z_c1
  c2 列: [0.8, 0.2, 0.3, 0.7] → 离线算好 s_c2, z_c2
  c3 列: [0.3, 0.7, 0.1, 0.2] → 离线算好 s_c3, z_c3

推理时，每个新来的 K 直接按列量化：
  K_new = [0.6, 0.3, 0.9, 0.4]
  c0: round((0.6 - z_c0) / s_c0)
  c1: round((0.3 - z_c1) / s_c1)
  ...
```

**不需要分组，不需要余留部分，直接量化。** 但前提是有离线校准集。

---

###### 四、技巧 2：在 RoPE 之前对 Key 量化

答主原文：

> 在旋转位置嵌入之前量化 Key 激活，以减轻其对量化的影响。

**RoPE（旋转位置编码）** 会改变 Key 的数值分布，让 Key 变得"更难量化"。

###### 用数据看

```
RoPE 之前的 Key（原始）：
K = [0.5, 0.2, 0.8, 0.3]
  max=0.8, min=0.2 → 范围小，容易量化

RoPE 之后的 Key（旋转后，示意值）：
K_rotated = [0.42, 0.35, 0.76, -0.28]
  max=0.76, min=-0.28 → 范围变大了，还出现了负数！
  → 更难量化了（需要更大的范围覆盖）
```

**KVQuant 的做法：**

```
在 RoPE 之前量化 Key：
  K_original = [0.5, 0.2, 0.8, 0.3] → 量化到 INT2 → 存起来

反量化时：
  K_int2 → 反量化 → K_recon_fp16 → 应用 RoPE → K_rotated
```

但 RoPE 在 Attention 计算时是必须要做的，所以 KVQuant 需要 **一个融合 Kernel**：

> 开发了一个融合 Kernel，以在反量化后高效应用 RoPE。

---

###### 五、技巧 3：非均匀 KV 缓存量化

答主原文：

> 非均匀 KV 缓存量化，通过导出每层敏感度加权的非均匀数据类型，以更好地表示分布。

**KVQuant 对不同层使用不同的量化数据类型**，不是所有层都用同一个 INT2 格式。

```
Layer 1 的 Key 分布：偏正态，中间密两边疏 → 用非均匀量化
Layer 10 的 Key 分布：偏均匀，到处都有 → 用均匀量化（INT2）
Layer 20 的 Key 分布：有离群值 → 用非均匀 + 特殊处理
```

**怎么确定每层用什么数据类型？** 通过离线校准集统计每层 Key 的分布，然后选择最匹配的数据类型。

---

###### 六、技巧 4：按向量密集和稀疏量化

答主原文：

> 按向量（Per-Vector）密集和稀疏量化，分别隔离每个向量的异常值，以最大限度地减少量化范围中的偏差。

**对每个向量（每个 token 的 K 或 V），检测是否有异常值，然后分别处理：**

```
K_step3 = [0.8, 5.0, 0.3, 0.1]  ← c1=5.0 是异常值

密集部分（正常值）：[0.8, 0.3, 0.1] → INT2 量化
稀疏部分（异常值）：[5.0] → 单独存为 FP16（或者单独量化）

这样密集部分的 scale 不会被 5.0 拉大，精度更好。
```

这和 LLM.int8() 的思路有点像——**把异常值隔离出来单独处理**。

---

###### 七、技巧 5：Attention Sink 感知量化

答主原文：

> 由于 Attention Sink 现象，模型对第一个 Token 中的量化误差异常敏感。通过仅保留第一个 Token 为 FP16 来确保模型的精度。

**第一个 token 不量化，保持 FP16：**

```
K_<s>: [0.5, 0.2, 0.8, 0.3]  ← FP16，不量化
K_爱:  [0.4, 0.6, 0.2, 0.7]  ← INT2，量化
K_吃:  [0.8, 5.0, 0.3, 0.1]  ← INT2，量化
K_苹果: [0.3, 0.5, 0.7, 0.2]  ← INT2，量化
```

**为什么？** 因为所有 token 在 Attention 时都会关注第一个 token，第一个 token 的量化误差会被放大到整个序列。

---

###### 八、技巧 6：离线计算缩放因子

答主原文：

> 在逐通道量化中，更新在线缩放因子是具有挑战性的，因为每个传入通道对应的缩放因子可能需要在 KV 缓存中添加新 Token 时进行更新。因此，离线（即在运行推理之前使用校准数据）计算统计数据。

**如果在线算 scale，问题如下：**

```
step1: K₁ → c0 列 = [0.5] → s_c0 = 0.5/3 = 0.167
step2: K₂ → c0 列 = [0.5, 0.4] → s_c0 需要更新！
step3: K₃ → c0 列 = [0.5, 0.4, 0.8] → s_c0 又要更新！
step4: K₄ → c0 列 = [0.5, 0.4, 0.8, 0.3] → s_c0 还要更新！
```

**每来一个新 token，之前所有已量化的数据都要重新量化一遍——这显然不行。**

**KVQuant 的解决方案：离线用校准集算好 scale，推理时固定，不更新。**

```
离线算好：
  c0 列: s_c0 = 0.8/3 = 0.267（从校准集统计的 max=0.8）

推理时每个新来的 K_c0 都用这个固定的 s_c0 量化：
  step1: 0.5 → round(0.5/0.267) = 2 → OK
  step2: 0.4 → round(0.4/0.267) = 1 → OK
  step3: 0.8 → round(0.8/0.267) = 3 → OK
  step4: 0.3 → round(0.3/0.267) = 1 → OK
  不需要重新量化已有的数据
```

**对于 Value 的逐 token 量化：** 因为每个 token 的 V 是独立算 scale 的，不受后续 token 影响，所以 Value 可以**在线**算 scale。

---

###### 九、6 个技巧分别解决什么问题

| 技巧 | 解决的问题 | 我们的数据中体现 |
|:----|:----------|:--------------|
| 1. Per-channel 量化 Key | Key 的通道性离群值 | c1 列有 5.0，按通道量化可以单独处理 |
| 2. RoPE 前量化 Key | RoPE 改变分布加大量化难度 | 旋转后值范围变大、出现负数 |
| 3. 非均匀量化 | 不同层分布不同 | 有些层偏正态，有些层偏均匀 |
| 4. Per-vector 密集+稀疏 | 单行内的异常值 | step3 的 c1=5.0 单独拎出来 |
| 5. Attention Sink 感知 | 第一个 token 的误差被放大 | 第一个 token 保持 FP16 |
| 6. 离线 scale | 在线更新需要重新量化已有数据 | 推理时直接用固定 scale |

---

###### 十、KVQuant vs KIVI

| | KIVI | KVQuant |
|:--|:----|:--------|
| 量化方式 | 在线量化（逐组） | **离线量化**（用校准集） |
| 是否需要分组 | 需要（G 个一组） | **不需要** |
| 是否需要余留部分 | 需要 | **不需要** |
| 是否需要校准集 | 不需要 | **需要** |
| Key 量化 | per-channel（组内） | per-channel（离线统算） |
| Value 量化 | per-token | per-token（在线） |
| 特殊技巧 | — | RoPE 前量化、非均匀量化、密集+稀疏、Attention Sink |

**KIVI 和 KVQuant 的核心区别就是：KIVI 在线做，KVQuant 离线做。** KIVI 不需要校准集但要分组，KVQuant 需要校准集但不需分组。

---

###### 十一、一句话总结

> **KVQuant 是离线 KV Cache 量化方案，把 6 个技巧组合在一起：Per-channel 量化 Key + RoPE 前量化 + 非均匀数据类型 + 隔离异常值 + Attention Sink 保 FP16 + 离线算 scale。不需要分组，不需要余留部分，但依赖校准集。**

## 2.14 Atom

**特点**：W4A4KV4、对异常值应用 INT8 量化

为了在保持准确性的同时将 LLMs 量化到极低比特精度，Atom 结合了一系列针对 LLM 特性量身定制的量化机制：

1. **混合精度量化与通道重排序**：对权重和激活进行混合精度量化，提高量化准确性。由于 8 位（FP8 和 INT8）足以表示异常值，因此对异常值应用 INT8 量化。
2. **细粒度分组量化**：将矩阵分成子组并在每个子组内独立进行量化，在准确性提升和反量化开销之间提供权衡。
3. **动态量化**：为每次推理中的每个激活矩阵定制量化参数。将量化操作融合到先前的算子中（类似 ZeroQuant）。采用具有精心选择的裁剪阈值进行对称量化。在量化权重矩阵时引入 GPTQ（离线过程，不牺牲运行时效率）。
4. **KV 缓存量化**：以注意力头的粒度应用非对称低比特量化，保持高精度。

### 用数据看混合精度

```
W 中离群值 6.0 → 用 INT8 存（8bit 足以表示异常值）
W 中其他值 0.5, 0.1, 0.3 → 用 INT4 存

X 中离群值 7.0 → 用 INT8 存
X 中其他值 → 用 INT4 存
```

---

## 2.15 QuaRot

**特点**：W4A4KV4

由于量化激活很困难（因为它们有较大的异常值），以前的工作依赖于使用校准集来表征异常特征，并在推理期间将它们保留在更高的精度。而 QuaRot 通过使用**随机 Hadamard 变换旋转模型输入**来解决异常特征问题。

作者利用计算不变性的思想，将 Hadamard 变换融合到权重矩阵中，从而得到一个没有异常特征的等效网络。这使得权重、激活和 KV 缓存可以量化到 4 比特，而模型精度损失很小。

### 两个阶段

1. **第一阶段**：对模型权重进行运算（以全精度），并在模型的前向传播中插入两个额外的 Hadamard 运算，有效地抑制异常值通道。
2. **第二阶段**：使用某种现有方法（GPTQ、RTN 等）量化权重，并在前向传播中添加量化操作以实现激活的在线量化。

默认情况下，使用 GPTQ 量化权重，而激活则使用简单的 RTN 方案进行在线量化。

### 用数据看 Hadamard 旋转

> **Hadamard 矩阵**：一种方块矩阵，元素仅包含 1 或 -1，不同的行具备正交性质。乘以 Hadamard 矩阵相当于对输入做"旋转"，将异常值分散到各个维度，消除离群通道。

```
Hadamard 矩阵 H₂ = [1, 1; 1, -1] 对 2 维向量做旋转

X = [2.0, 0.5, 0.8, 0.3]
每两个元素做 Hadamard 变换：
  [2.0, 0.5] × H₂ = [2.0+0.5, 2.0-0.5] = [2.5, 1.5]   ← 离群值被"稀释"了
  [0.8, 0.3] × H₂ = [0.8+0.3, 0.8-0.3] = [1.1, 0.5]

旋转后 X' = [2.5, 1.5, 1.1, 0.5]  ← 没有明显的离群值了！
```

---

## 2.16 QoQ

**特点**：W4A8KV4

为了实现 W4A8KV4 量化精度的理论吞吐量优势，同时不牺牲大语言模型的有效性，QoQ 算法采用渐进式分组量化、SmoothAttention 和各种通用量化优化功能。

### 渐进式分组量化

给定权重张量，首先**应用逐通道（per-channel）对称 INT8 量化**，然后进一步在中间权重张量上**应用逐组（per-group）非对称 INT4 量化**。

```
用数据推演（out0 行 [0.5, 0.1, 0.3, 6.0]）：
  Step 1: per-channel INT8 → s = 6.0/127 = 0.0472 → [11, 2, 6, 127]
  Step 2: per-group INT4（每组 2 个元素）→ [11,2]→s=11/7=1.57→[7,1]；[6,127]→s=127/7=18.1→[0,7]
```

对于 W4A8 GEMM 计算：4 位量化权重张量先反量化为中间 8 位量化权重张量，然后执行 INT8 矩阵乘法（就像 W8A8 逐通道量化一样）。引入保护范围，允许将反量化操作融合到 W4A8 GEMM Kernel 中，实现全寄存器级并行。

### SmoothAttention

由于 Key 矩阵在每个 Attention 头中都有固定的异常值通道，受 SmoothQuant 启发，提出**通过逐通道因子缩小 Key 缓存中的异常通道**。将缩放因子融合到前一层的权重中，消除额外 Kernel 调用开销。

### LLM 量化通用优化

- 块输入模块旋转（受 QuaRot、Quip 启发）
- 块输出模块平滑（受 SmoothQuant 启发）
- 激活感知的通道重排序（使得具有相似显著性的通道在同一个量化组，降低精度损失）
- 权重裁剪（减少量化误差）

---

## 2.17 方法总回顾：18 个方法的核心思想一句话总结

```
RTN:         四舍五入，最简单，精度最差
LLM-QAT:     训练时模拟量化，让模型提前适应
QLoRA:       量化后加 LoRA 补偿
LLM.int8():  INT8/FP16 混合精度，离群值单独处理
GPTQ:        量化后调整其他权重来补偿，优化视角
SmoothQuant: 平滑因子把量化难度从激活转移到权重
AWQ:         保护显著通道，按重要性分配精度
SpQR:        隔离离群权重 + 双层量化
ZeroQuant:   分组权重 + 逐 token 动态激活
ZeroQuant-V2:加低秩矩阵补偿误差
ZeroQuant-FP:FP4/FP8 + 位移加速
FP6-LLM:    TC-FPx kernel 支持 6bit
KIVI:        KV 2bit，Key 按通道/Value 按 token
IntactKV:    关键词元 KV 无损
KVQuant:     多种技巧组合优化 KV 量化
Atom:        W4A4KV4，混合精度处理异常值
QuaRot:      Hadamard 旋转消除异常值
QoQ:         W4A8KV4，渐进式分组量化
```

> 🔍 **方法家族划分（便于记忆）**
>
> 这 18 个方法按"如何处理离群值/敏感权重"可分四条路线：
> - **路线 1：直接量化（RTN）**——什么都不处理，精度最差；
> - **路线 2：抑制离群值（数学等价变换）**——SmoothQuant、AWQ、QuaRot、QoQ（旋转/平滑）、Atom（异常值 INT8）；
> - **路线 3：优化补偿（量化后调整）**——GPTQ、ZeroQuant-V2（LoRC）；
> - **路线 4：混合精度隔离**——LLM.int8()、SpQR；
> - **训练/微调路线**——LLM-QAT（训练模拟量化）、QLoRA（微调补偿）；
> - **KV 专用路线**——KIVI、IntactKV、KVQuant。
>
> 主线只有一句话：**量化误差主要来自离群值拉大步长，所有方法都在回答"怎么让 step 别被 outlier 拖垮"。**

---

# 第三部分：量化对大模型的影响

> 该部分参考：**Evaluating Quantized Large Language Models**

## 3.1 量化对模型精度的影响

- **越大的模型对于权重和 KV Cache 量化容忍度更高，而对权重-激活值量化容忍度更低**。原因：模型越大，分布在权重和 KV Cache 中的离群值越少，而分布在激活值中的离群值越多。
- 小模型的 INT4 仅权重量化会导致精度大幅下降，但这种影响会随着模型大小的增加而减弱。
- INT8 激活导致小型模型的准确度下降最小，但较大的模型表现出更大的下降。
- 利用专家混合（MoE）技术增加模型大小并不一定增强模型对量化的容忍度。
- 在大多数任务上，大部分 LLMs 可以通过 W4 或 KV4 量化保持其性能。当将 LLMs 量化到 W3 或 KV3 时，所有模型家族中小型模型的性能都会明显下降。此外，对于 W2 或 KV2，大多数模型都会经历显著的性能损失。
- 对于权重-激活量化，W4A8 可以达到良好性能。W4A4 量化将严重影响 LLMs 的效果。
- 短文本任务：W8KV4 几乎无损失。长文本任务：W8KV8 是更好的选择。
- 在具有大批量大小和长文本的实际应用中，一种常见做法是**同时量化权重和 KV 缓存**（因为两者都吃显存，且都有量化容忍度；激活保持高精度或采用 W4A8 等中间方案）。

> 🔍 **模型大小 vs 量化容忍度**
>
> ```
>             权重W量化     KV Cache量化     激活A量化
> 小模型(7B)    ❌敏感       ❌敏感          ✅还行
> 大模型(70B)   ✅不敏感      ✅不敏感         ❌敏感
> ```
> 大模型的权重分布更"平滑"，离群值少，所以 W4 也能保住精度；但大模型的激活离群值更多、更大，所以 A8 可能都不够，需要 SmoothQuant 或 QuaRot 来处理。MoE 模型的量化不是"模型更大就更好量化"，因为 MoE 的路由机制对精度更敏感。

## 3.2 量化对不同任务类型能力的影响

### 自然语言处理能力

对于多数自然语言处理任务，大多数大模型可以采用 W4、W4A8、KV4、W8KV4 量化位宽，几乎没有性能损失（<2%）。当使用极低的位宽时（W2 或 W4A4），若想使用 SOTA 量化方法将模型性能恢复到类似于 FP16 的水平是一个很大的挑战。

### 涌现能力

四类涌现能力对于量化的容忍度由高到低：
```
上下文学习 ~ 指令跟随 > 多步推理 ~ 自我校准
```

- **上下文学习**：适当增加例子数量（由 0 到 5）可以提升性能；但例子过多（~10 个）时性能不会持续提升，甚至略有下降。
- **多步推理**：**数学推理对于量化的容忍度低于常识推理**。量化后会出现抄写错误、漏看条件、计算错误、逻辑错误。
- 对于较小的模型（<13B），W8 或 KV8 适合维持多步推理和自我校准能力。
- 对于较大的模型（>13B），W4 或 KV4 适合维持涌现能力。
- **W4A8 不会导致显著的性能损失，但 W4A4 会导致所有模型完全丧失涌现能力。**

### 对话能力

- W8、W8A8、KV4：对话能力几乎不损失。
- W3、KV3：输出出现语句重复、无意义符号。
- W2、W4A4、KV2：输出出现词语重复，有时输出随机词语。

### 长文本能力

- 相较于短文本（<4k），输入长文本（>4k）时模型性能对权重和 KV Cache 量化容忍度更低。
- 多数模型对 KV Cache 量化的容忍度低于对权重、权重-激活量化。
- 推荐使用 W4、W4A8、KV8 量化位宽执行长文本任务。

### 一张表总结

| 场景 | 推荐位宽 | 原因 |
|:----|:--------|:----|
| 通用场景 | W4 / KV4 | 大多数任务无显著损失 |
| 小模型（<13B）多步推理 | W8 / KV8 | 小模型对量化更敏感 |
| 大批量推理 | FP8 / W4A8 | 计算瓶颈，需要 INT8 加速 |
| 长文本 | W8KV8 / W4A8KV8 | KV Cache 对长文本更敏感 |
| 对话 | W8 / KV4 | 无损 |
| 涌现能力要求高 | W4A8 | W4A4 会完全丧失涌现能力 |

## 3.3 量化对于模型推理性能的影响

- **Weight-only 量化可以显著加速 decoding 阶段**，从而改善端到端延迟。
- 关于 prefill 阶段，Weight-only 量化可能实际上会**增加延迟**。
- 随着批量大小和输入长度的增加，Weight-only 量化所带来的加速效果逐渐减小。
- 对于较大的模型，Weight-only 量化提供了更大的益处，因为较大模型尺寸的内存访问开销显著增加。

> 🔍 **为什么 decoding 加速、prefill 反而变慢？**
>
> - **Decoding 阶段**是 memory-bound（内存带宽瓶颈），主要时间花在把权重从显存搬到计算单元。W4 的搬运量是 FP16 的 1/4，所以快。
> - **Prefill 阶段**是 compute-bound（计算瓶颈），反量化（INT4→FP16）多了一步操作，所以可能更慢。
> - batch=1：内存带宽瓶颈，W4 加速明显；batch=16：计算和内存都瓶颈，加速减小；batch=64：计算瓶颈为主，W4 可能没有加速。

> **一句话总结**：量化对大模型的影响 = 模型越大对 W/KV 越不敏感但对 A 越敏感 + 涌现能力 > 基本能力 > 对话能力 + Weight-only 加速 decoding 但对 prefill 没帮助。

---

**博主原文配图**（源自知乎原文）：

![原文图 15](https://picx.zhimg.com/v2-1221c89b0ef07270d3f8691f5c059705_r.jpg?source=2c26e567)


---

### 📌 原始 Q&A：量化对大模型的影响（历史第 80 轮）

> 以下为本节知识对应的**历史对话原文**（Q = 当时你的提问，A = 当时的完整回答）。已剔除 IDE 操作等无关行，内容原样保留；其中推演与正文讲解或有重复，作为复习回顾用。

##### Q（历史第 80 轮）

量化对大模型的影响

##### A（历史第 81 轮，正文原样保留；仅将内部标题降级以保持目录层级）

好！这部分不用 4×4 数据了，因为讲的是 **宏观影响规律**。我用表格和对比的方式让你一目了然。

---

###### 一、量化对模型精度的影响

答主原文给出了几个关键发现：

###### ① 越大的模型，对权重和 KV Cache 量化容忍度越高，对激活量化越不 tolerant

```
模型大小 vs 量化容忍度：

            权重W量化     KV Cache量化     激活A量化
小模型(7B)    ❌敏感       ❌敏感          ✅还行
大模型(70B)   ✅不敏感      ✅不敏感         ❌敏感
```

**为什么？** 答主原文解释：

> 模型越大，分布在权重和 KV Cache 中的离群值越少，而分布在激活值中的离群值越多。

大模型的权重分布更"平滑"，离群值少，所以 W4 也能保住精度。
但大模型的激活离群值更多、更大，所以 A8 可能都不够，需要 SmoothQuant 或 QuaRot 来处理。

---

###### ② 各种位宽的精度表现

```
W4（仅权重量化）：✅ 小模型精度下降明显，大模型影响小
W8A8：              ✅ 小模型 OK，大模型下降大
W4A8：              ✅ 大多数模型表现良好
W4A4：              ❌ 严重影响大部分模型
KV4：               ✅ 大多数任务 OK
KV2：               ❌ 大多数模型性能显著下降
W2：                ❌ 大多数模型性能显著下降
```

**关键结论**：

> 在大多数任务上，大部分 LLMs 可以通过 **W4 或 KV4** 量化保持其性能。

> 对于权重-激活量化，**W4A8** 可以达到良好性能。**W4A4 量化将严重影响 LLMs 的效果**。

---

###### ③ MoE 的特殊性

> 利用专家混合（MoE）技术增加模型大小并不一定增强模型对量化的容忍度。

MoE 模型的量化不是"模型更大就更好量化"，因为 MoE 的路由机制对精度更敏感。

---

###### 二、量化对不同任务类型能力的影响

###### ① 基本 NLP 能力（语言建模、理解、生成）

> 对于多数自然语言处理任务，大多数大模型可以采用 **W4、W4A8、KV4、W8KV4** 量化位宽，几乎没有性能损失（<2%）。

> 当使用极低位宽时，如 **W2 或 W4A4**，想恢复到 FP16 水平是很大的挑战。

---

###### ② 涌现能力

答主原文给出容忍度排序：

> 上下文学习 ≈ 指令跟随 > 多步推理 ≈ 自我校准

```
高容忍度 ───────────────────────────────────→ 低容忍度
上下文学习 ≈ 指令跟随    >    多步推理 ≈ 自我校准
    W4 够用                      W8/KV8 才够
```

**上下文学习（In-Context Learning）**：给几个例子就能学。量化影响小。

**指令跟随（Instruction Following）**：听懂指令去执行。量化影响小。

**多步推理（Multi-step Reasoning）**：数学题、逻辑推理。量化影响大。

**自我校准（Self-calibration）**：自己检查自己的输出对不对。量化影响最大。

**具体来说：**

> 数学推理对于量化的容忍度低于常识推理。

数学推理出错类型包括：抄写错误、漏看条件、计算错误、逻辑错误。量化后这些错误会增多。

> 对于较小的模型（小于 13B），W8 或 KV8 适合于维持多步推理和自我校准能力。

> 对于较大的模型（大于 13B），W4 或 KV4 适合于维持涌现能力。

> W4A8 不会导致显著的性能损失，但 **W4A4 会导致所有模型完全丧失涌现能力**。

---

###### ③ 对话能力

```
W8、W8A8、KV4：   对话能力几乎不损失 ✅
W3、KV3：         语句重复、无意义符号 ❌
W2、W4A4、KV2：   词语重复、随机词语 ❌❌
```

---

###### ④ 长文本能力

> 相较于短文本（<4k），输入长文本（>4k）时模型性能对权重和 KV Cache 量化容忍度更低。

> 对于长文本任务，多数模型对 KV Cache 量化的容忍度低于对权重、权重-激活量化。

**推荐：**

```
短文本：W8KV4 几乎无损失
长文本：W8KV8 是更好的选择
        推荐 W4、W4A8、KV8
```

---

###### 三、量化对于模型推理性能的影响

###### ① Decoding 阶段（生成 token）

> Weight-only 量化可以显著加速 decoding 阶段，从而改善端到端延迟。

Decoding 阶段是 **memory-bound**（内存带宽瓶颈）。主要时间花在把权重从显存搬到计算单元上。W4 的搬运量是 FP16 的 1/4，所以快。

###### ② Prefill 阶段（处理输入）

> 关于 prefill 阶段，Weight-only 量化可能实际上会增加延迟。

Prefill 阶段是 **compute-bound**（计算瓶颈）。反量化（INT4→FP16）多了一步操作，所以可能更慢。

###### ③ Batch size 和输入长度的影响

> 随着批量大小和输入长度的增加，Weight-only 量化所带来的加速效果逐渐减小。

batch=1：内存带宽瓶颈，W4 加速明显 🚀
batch=16：计算和内存都瓶颈，加速减小
batch=64：计算瓶颈为主，W4 可能没有加速

###### ④ 模型大小的影响

> 对于较大的模型，Weight-only 量化提供了更大的益处，因为较大模型尺寸的内存访问开销显著增加。

70B 模型用 W4 省显存的效果远大于 7B 模型。

---

###### 四、一张表总结

| 场景 | 推荐位宽 | 原因 |
|:----|:--------|:----|
| 通用场景 | W4 / KV4 | 大多数任务无显著损失 |
| 小模型（<13B）多步推理 | W8 / KV8 | 小模型对量化更敏感 |
| 大批量推理 | FP8 / W4A8 | 计算瓶颈，需要 INT8 加速 |
| 长文本 | W8KV8 / W4A8KV8 | KV Cache 对长文本更敏感 |
| 对话 | W8 / KV4 | 无损 |
| 涌现能力要求高 | W4A8 | W4A4 会完全丧失涌现能力 |

---

###### 五、一句话总结

> **量化对大模型的影响 = 模型越大对 W/KV 越不敏感但对 A 越敏感 + 涌现能力 > 基本能力 > 对话能力 + Weight-only 加速 decoding 但对 prefill 没帮助。**

# 第四部分：如何选择正确的量化方法（最佳实践）

> 以下是 TensorRT-LLM 中选择量化方法的最佳实践。

量化方法包含三个主要组成部分：
- 权重精度格式
- 激活精度格式
- 校准算法

## 按批量大小选择

### 小批量推理（batch ≤ 4）：内存限制

推理通常是"内存限制"，吞吐量受权重加载时间限制。因此，**仅权重量化方法**可提供不错的性能提升：
- INT4 AWQ（W4A16）
- INT4-FP8 AWQ（W4A8）
- FP6-LLM（W6A16）

### 大批量推理（batch ≥ 16）：内存带宽 + 计算密度

建议选择**权重和激活同时量化**以及使用较低精度计算 Kernel 的方法：
- 优先使用 **FP8**（几乎不会导致精度下降，性能强大）
- 如果 FP8 性能不够，尝试 **INT4-FP8 AWQ**
- 如果部署在 Ampere GPU 或更早版本，使用 **INT4 AWQ** 或 **INT8 SmoothQuant**

## 量化方法对比表

| 量化方法 | 小批量性能 | 大批量性能 | 精度下降 | 详情 |
|---|---|---|---|---|
| **FP8** | 中 | 中 | 非常低 | min-max 校准，FP8 per-tensor 权重和激活量化，压缩至 50%，校准时间分钟级，支持 Ada/Hopper+ |
| **INT8 SmoothQuant** | 中 | 中 | 中 | SmoothQuant 校准变体，8bit 整数，逐通道权重+逐张量激活，压缩至 50%，校准时间分钟级，大多数 GPU 支持 |
| **INT4 Weights only AWQ (W4A16)** | 高 | 低 | 低 | AWQ 校准，4bit 仅权重量化（group-wise），压缩至 25%，校准时间数十分钟，支持 Ampere+ |
| **INT4-FP8 AWQ (W4A8)** | 高 | 中 | 低 | AWQ 校准 4bit 权重 + FP8 激活（per-tensor），压缩至 25%，校准时间数十分钟，支持 Ada/Hopper+ |

> 🔍 **博主原文补充**：上表中的性能和影响是在 **10 多个流行的 LLMs** 上测量而来的；FP8/INT8 SmoothQuant 通过 TensorRT、TensorRT-LLM 进行部署（压缩至 50%），INT4 系列通过 TensorRT-LLM 部署（压缩至 25%）。根据特定的使用场景，用户可能对精度下降和校准时间有不同的容忍度，可按上表权衡。

---

# 第五部分：其他答主补充

## Ironboy（获 207 赞）

> SOTA 这个东西跟时间相关，做的比较早的那些比如 GPTQ，SmoothQuant，AWQ 就不提了，新一点的只说开源的吧。

**2025 年 3 月 16 日更新：**

- **PTQ**：OmniQuant，QLLM，DuQuant，FlatQuant，SpinQuant，PrefixQuant。目前看起来后面这几个带旋转矩阵的效果最好，但需要考虑额外的部署开销。
- **QAT**：QLoRA，LoftQ，EfficientQAT，ParetoQ（目前没开源）。

**小众赛道：**
- **向量量化**：AQLM，QUIP，QUIP#
- **1bit/1.58bit 量化**：BitNet，OneBit，BiLLM，STB-LLM，FBI-LLM

## 不归牛顿管的熊猫

> 首先，问题就不成立。量化领域没有 SOTA 一说，不同场景的不同模型适用的量化方案都不尽相同，只有试了才知道，凡是称自己是 SOTA 量化方案，都是扯淡。

该答主详细介绍了 **any4**（Meta FAIR 实验室 2025 年 ICML 推出的 4bit 量化方案）：

### any4 核心思想

给权重矩阵的每一行，学一个专属的 16 个浮点数查找表（LUT）。
- 每个权重最终只存一个 4bit 索引
- 推理时通过索引去 LUT 里取高精度浮点值
- 一行权重 = 一组 4bit 索引 + 一个 16 个 FP16/BF16 的 LUT

### any4 vs 传统 4bit 量化

| 对比维度 | any4 | INT4 | FP4 | NF4 |
|---|---|---|---|---|
| 核心定位 | LLM 专用学习型 4bit | 传统均匀整数量化 | 极简浮点格式 | 预设高斯分布浮点 |
| 实现方式 | 分组缩放 + 逐行加权 K-Means 学习码本 + LUT 查表 + 单样本校准 | 线性均匀缩放 + 固定整数映射 | 符号位+指数位+尾数位固定解析 | 预定义 16 个高斯拟合值 + 全局固定查表 |
| 校准成本 | 仅需 1 条多样化样本 | 无需/少量样本 | 无需 | 需数据集校准 |
| 精度表现 | 最优，大组宽下依然稳定 | 一般 | 最差 | 较好 |
| 推理速度 | 快，与 NF4 持平 | 最快 | 快 | 快 |
| 内存开销 | 极低，仅增加 0.0625bit/entry | 低 | 低 | 低 |

### any4 适用场景

适合追求 4bit 轻量化高精度、希望简化工程流程、校准数据有限或主要面向对话类小批量推理的开发者。即便将量化组宽扩大至 1024，仍能保持稳定精度。

不适合：极致追求推理速度、可牺牲精度、运行在不支持浮点与查表操作的低端硬件上、不愿编译 CUDA 算子的场景。2bit/3bit 极限压缩场景下也非最优方案。

---

# 附录：参考文档

## 博主本系列量化文章索引（原回答开篇列出的学习路径）

- [大模型量化概述](https://www.zhihu.com/question/627484732/answer/3261671478)
- 量化感知训练：
  - [大模型量化感知训练技术原理：LLM-QAT](https://zhuanlan.zhihu.com/p/647589650)
  - [大模型量化感知微调技术原理：QLoRA](https://zhuanlan.zhihu.com/p/636215898)
- [训练后量化](https://zhida.zhihu.com/search?content_id=710333991&content_type=Answer&match_order=1&q=%E8%AE%AD%E7%BB%83%E5%90%8E%E9%87%8F%E5%8C%96&zhida_source=entity)：
  - [大模型量化技术原理：GPTQ、LLM.int8()](https://zhuanlan.zhihu.com/p/680212402)
  - [大模型量化技术原理：SmoothQuant](https://www.zhihu.com/question/576376372/answer/3388402085)
  - [大模型量化技术原理：AWQ、AutoAWQ](https://zhuanlan.zhihu.com/p/681578090)
  - [大模型量化技术原理：SpQR](https://zhuanlan.zhihu.com/p/682871823)
  - [大模型量化技术原理：ZeroQuant系列](https://link.zhihu.com/?target=https%3A//juejin.cn/post/7338284106797432873)
  - [大模型量化技术原理：FP8](https://www.zhihu.com/question/658712811/answer/3596678896)
  - [大模型量化技术原理：FP6](https://link.zhihu.com/?target=https%3A//juejin.cn/post/7412893752090853386)
  - [大模型量化技术原理：KIVI、IntactKV、KVQuant](https://zhuanlan.zhihu.com/p/5932153295)
  - [大模型量化技术原理：Atom、QuaRot](https://zhuanlan.zhihu.com/p/6281447174)
  - [大模型量化技术原理：QoQ量化及QServe推理服务系统](https://zhuanlan.zhihu.com/p/8047106486)
  - [大模型量化技术原理：总结](https://www.zhihu.com/question/10439431486)

## 本文参考文档

- [量化方法怎么选？如何评估量化后的大模型 LLM？](https://zhuanlan.zhihu.com/p/695144724)
- [Qllm-Eval：大模型压缩量化方案怎么选？看这一篇就够了](https://zhuanlan.zhihu.com/p/704228271)
- [Evaluating Quantized Large Language Models](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2402.18158)
- [Best practices to choose the right quantization methods](https://link.zhihu.com/?target=https%3A//nvidia.github.io/TensorRT-Model-Optimizer/guides/_choosing_quant_methods.html)
- [量化粒度](https://link.zhihu.com/?target=https%3A//medium.com/%40curiositydeck/quantization-granularity-aec2dd7a0bb4)
- [量化时机与粒度](https://link.zhihu.com/?target=https%3A//blog.gopenai.com/model-quantization-3-timing-and-granularity-a0978c6e58d4)
- [what's the meaning of "Groupwise 4-bit (128)" #3559](https://link.zhihu.com/?target=https%3A//github.com/pytorch/executorch/issues/3559)
- [揭秘NVIDIA大模型推理框架：TensorRT-LLM](https://link.zhihu.com/?target=https%3A//hub.baai.ac.cn/view/34855)

---

> **本文说明**：本文基于知乎博主「吃果冻不吐果冻皮」的主回答整理，融合了学习过程中 44 轮深度追问的关键澄清点、全部数值推演与易错校正。`🔍 学习追问` 与 `⚠️ 易错校正` 部分记录了当时反复确认、容易混淆的知识点及其最终结论，可作为复习时的重点关注项。
