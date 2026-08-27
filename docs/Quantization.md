# Quantization：量化基础 + GPTQ / AWQ / SmoothQuant / KIVI / KVQuant

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
